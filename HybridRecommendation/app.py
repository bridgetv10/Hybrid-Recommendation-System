"""
Music Recommendation Web App
Enter 5 favorite artists, adjust exploration, get recommendations
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import pickle
import os

from complete_integrated_recommender import IntegratedRecommender

app = Flask(__name__)

# Global variables for the trained system
recommender_system = None
artists_df = None
user_artists_df = None
available_artists = None


def load_or_train_system():
    """Load trained system or train if not exists"""
    global recommender_system, artists_df, user_artists_df, available_artists

    model_path = Path('trained_models')
    model_path.mkdir(exist_ok=True)

    print("Loading data...")
    base_path = Path('lastfm_data')

    user_artists_df = pd.read_csv(base_path / "user_artists.dat", sep='\t', encoding='latin-1')

    try:
        artists_df = pd.read_csv(base_path / "artists.dat", sep='\t', encoding='utf-8', on_bad_lines='skip')
    except:
        artists_df = pd.read_csv(base_path / "artists.dat", sep='\t', encoding='latin-1', on_bad_lines='skip')

    tags = pd.read_csv(base_path / "tags.dat", sep='\t', encoding='latin-1')
    user_taggedartists = pd.read_csv(base_path / "user_taggedartists.dat", sep='\t', encoding='latin-1')

    # Get available artists (those in the system)
    available_artists = artists_df[['id', 'name']].copy()
    available_artists.columns = ['artistID', 'name']
    available_artists = available_artists.sort_values('name')

    print(f"Loaded {len(available_artists)} artists")

    # Check if model exists
    model_file = model_path / 'recommender_system.pkl'

    if model_file.exists():
        print("Loading pre-trained system...")
        try:
            with open(model_file, 'rb') as f:
                recommender_system = pickle.load(f)
            print("✓ System loaded!")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train new model...")

    # Train new system
    print("\n" + "="*80)
    print("TRAINING NEW SYSTEM (this will take ~15 minutes)")
    print("="*80)

    recommender_system = IntegratedRecommender(embedding_dim=64)
    recommender_system.load_data(user_artists_df, artists_df, tags, user_taggedartists)
    recommender_system.train(use_tag_clustering=True, ncf_epochs=10)

    # Save the system
    print(f"\nSaving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump(recommender_system, f)

    print("✓ System trained and saved!")


def create_synthetic_user(favorite_artist_ids):
    """
    Create a synthetic user profile based on favorite artists
    Returns a user_id that we can use for recommendations

    Improved version: Considers both overlap AND content similarity
    """
    favorite_set = set(favorite_artist_ids)

    best_match_user = None
    best_score = 0

    # Sample MORE users for better matching (3000 instead of 1000)
    all_users = user_artists_df['userID'].unique()
    sample_users = all_users[:min(3000, len(all_users))]

    # Get content similarity matrix if available
    use_content = (recommender_system is not None and
                   hasattr(recommender_system, 'artist_similarity') and
                   recommender_system.artist_similarity is not None)

    for user_id in sample_users:
        user_data = user_artists_df[user_artists_df['userID'] == user_id]
        user_artists = set(user_data['artistID'])

        # 1. Overlap score (number of favorites they also listen to)
        overlap = len(favorite_set & user_artists)

        if overlap == 0:
            continue

        # 2. Content similarity score
        content_score = 0.0
        if use_content:
            # Check if favorites have similar content to user's top artists
            user_top_artists = user_data.nlargest(10, 'weight')['artistID'].tolist()

            similarities = []
            for fav_artist in favorite_artist_ids:
                if fav_artist in recommender_system.artist_profiles['artistID'].values:
                    fav_idx = recommender_system.artist_profiles[
                        recommender_system.artist_profiles['artistID'] == fav_artist
                    ].index[0]

                    for user_artist in user_top_artists:
                        if user_artist in recommender_system.artist_profiles['artistID'].values:
                            user_idx = recommender_system.artist_profiles[
                                recommender_system.artist_profiles['artistID'] == user_artist
                            ].index[0]

                            sim = recommender_system.artist_similarity[fav_idx, user_idx]
                            similarities.append(sim)

            if similarities:
                content_score = np.mean(similarities)

        # Combined score: Overlap is important, but content similarity helps
        # Weight overlap more heavily (70/30 split)
        combined_score = (overlap * 0.7) + (content_score * 10 * 0.3)

        if combined_score > best_score:
            best_score = combined_score
            best_match_user = user_id

    return best_match_user


def get_recommendations_for_favorites(favorite_artist_names, exploration=0.5, n_recommendations=20):
    """
    Get recommendations based on favorite artists

    Improved version: Filters out recommendations with low content similarity
    """
    # Convert names to IDs
    favorite_ids = []
    matched_names = []

    for name in favorite_artist_names:
        matches = available_artists[
            available_artists['name'].str.lower() == name.lower()
        ]

        if len(matches) > 0:
            favorite_ids.append(matches.iloc[0]['artistID'])
            matched_names.append(matches.iloc[0]['name'])

    if len(favorite_ids) == 0:
        return {
            'error': 'No matching artists found',
            'matched': [],
            'recommendations': []
        }

    # Find similar user
    similar_user = create_synthetic_user(favorite_ids)

    # Get recommendations (get more than needed, we'll filter)
    try:
        # Request more candidates so we can filter
        retrieval_multiplier = 3 if exploration < 0.5 else 2
        recs = recommender_system.recommend(
            user_id=similar_user,
            n_recommendations=n_recommendations * retrieval_multiplier,
            exploration=exploration
        )

        # Calculate content similarity threshold
        # Lower exploration = stricter filtering (only similar genres)
        # Higher exploration = more lenient filtering (discover new genres)
        min_content_score = 0.15 - (exploration * 0.1)  # Range: 0.05 to 0.15

        # Format results with content filtering
        recommendations = []
        for rec in recs:
            # Skip if already in favorites
            if rec['artistID'] in favorite_ids:
                continue

            content_score = float(rec.get('content_score', 0.5))

            # Filter by content similarity (unless high exploration)
            if exploration < 0.7 and content_score < min_content_score:
                continue

            recommendations.append({
                'rank': len(recommendations) + 1,
                'name': rec['name'],
                'score': float(rec['score']),
                'content_score': content_score,
                'popularity': int(rec.get('popularity', 50))
            })

            # Stop once we have enough
            if len(recommendations) >= n_recommendations:
                break

        return {
            'matched': matched_names,
            'recommendations': recommendations[:n_recommendations],
            'exploration_level': exploration,
            'content_filter_threshold': min_content_score
        }

    except Exception as e:
        return {
            'error': str(e),
            'matched': matched_names,
            'recommendations': []
        }


def get_niche_recommendations_for_favorites(favorite_artist_names, n_recommendations=10,
                                           popularity_threshold=45.0):
    """
    Get niche artist recommendations based on favorite artists

    Improved version: Applies content filtering to niche recommendations too
    """
    # Convert names to IDs
    favorite_ids = []
    matched_names = []

    for name in favorite_artist_names:
        matches = available_artists[
            available_artists['name'].str.lower() == name.lower()
        ]

        if len(matches) > 0:
            favorite_ids.append(matches.iloc[0]['artistID'])
            matched_names.append(matches.iloc[0]['name'])

    if len(favorite_ids) == 0:
        return {
            'error': 'No matching artists found',
            'matched': [],
            'recommendations': []
        }

    # Find similar user
    similar_user = create_synthetic_user(favorite_ids)

    # Get niche recommendations (get more candidates for filtering)
    try:
        recs = recommender_system.recommend(
            user_id=similar_user,
            n_recommendations=n_recommendations * 3,  # Get 3x for filtering
            exploration=0.3
        )

        # Apply content filtering (slightly more lenient for niche)
        min_content_score = 0.08  # Lower threshold for niche discovery

        # Format results - filter for niche artists based on popularity threshold
        recommendations = []
        for rec in recs:
            # Skip if already in favorites
            if rec['artistID'] in favorite_ids:
                continue

            content_score = float(rec.get('content_score', 0.5))
            popularity = int(rec.get('popularity', 50))

            # Filter by popularity threshold (niche = less popular)
            if popularity > popularity_threshold:
                continue

            # Filter by content similarity
            if content_score < min_content_score:
                continue

            recommendations.append({
                'rank': len(recommendations) + 1,
                'name': rec['name'],
                'score': float(rec['score']),
                'content_score': content_score,
                'popularity': popularity
            })

            if len(recommendations) >= n_recommendations:
                break

        return {
            'matched': matched_names,
            'recommendations': recommendations[:n_recommendations],
            'popularity_threshold': popularity_threshold,
            'content_filter_threshold': min_content_score
        }

    except Exception as e:
        return {
            'error': str(e),
            'matched': matched_names,
            'recommendations': []
        }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/search_artists')
def search_artists():
    """Search for artists by name"""
    query = request.args.get('q', '').lower()
    
    if len(query) < 2:
        return jsonify([])
    
    # Search for matching artists
    matches = available_artists[
        available_artists['name'].str.lower().str.contains(query, na=False)
    ].head(20)
    
    results = [
        {'id': int(row['artistID']), 'name': row['name']}
        for _, row in matches.iterrows()
    ]
    
    return jsonify(results)


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get recommendations"""
    data = request.json
    
    favorite_artists = data.get('favorites', [])
    exploration = float(data.get('exploration', 0.5))
    n_recommendations = int(data.get('n_recommendations', 20))
    
    if len(favorite_artists) < 3:
        return jsonify({
            'error': 'Please select at least 3 favorite artists'
        }), 400
    
    results = get_recommendations_for_favorites(
        favorite_artists,
        exploration=exploration,
        n_recommendations=n_recommendations
    )
    
    return jsonify(results)


@app.route('/api/recommend_niche', methods=['POST'])
def recommend_niche():
    """Get niche artist recommendations"""
    data = request.json

    favorite_artists = data.get('favorites', [])
    n_recommendations = int(data.get('n_recommendations', 10))
    popularity_threshold = float(data.get('popularity_threshold', 45.0))

    if len(favorite_artists) < 3:
        return jsonify({
            'error': 'Please select at least 3 favorite artists'
        }), 400

    results = get_niche_recommendations_for_favorites(
        favorite_artists,
        n_recommendations=n_recommendations,
        popularity_threshold=popularity_threshold
    )

    return jsonify(results)


@app.route('/api/artist_stats')
def artist_stats():
    """Get stats about available artists"""
    return jsonify({
        'total_artists': len(available_artists),
        'total_users': len(user_artists_df['userID'].unique()),
        'total_listens': int(user_artists_df['weight'].sum())
    })




# ============================================================================
# HTML TEMPLATE
# ============================================================================

@app.route('/templates/index.html')
def serve_template():
    """Serve the HTML template"""
    return render_template('index.html')


if __name__ == '__main__':
    # Load or train system on startup
    print("\n" + "="*80)
    print("MUSIC RECOMMENDATION WEB APP")
    print("="*80)
    
    load_or_train_system()
    
    print("\n" + "="*80)
    print("✓ System ready!")
    print("="*80)
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:5001")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)