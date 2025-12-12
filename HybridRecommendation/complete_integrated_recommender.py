import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Tag clustering for artist profiles

class TagClusterer:
    
    def __init__(self, tags_df, user_taggedartists_df):
        self.tags = tags_df.copy()
        self.user_taggedartists = user_taggedartists_df.copy()
        self.tag_mapping = {}
        
    def analyze_tag_quality(self):
        print("\nAnalyzing tag quality...")
        tag_counts = self.user_taggedartists.groupby('tagID').size().sort_values(ascending=False)
        print(f"Found {len(tag_counts)} unique tags")
        print(f"Single-use tags: {(tag_counts == 1).sum()}")
        return tag_counts
    
    def normalize_tag_string(self, tag_str):
        if pd.isna(tag_str):
            return ""
        tag_str = str(tag_str).lower()
        tag_str = re.sub(r'[_\-]', ' ', tag_str)
        tag_str = re.sub(r'[^\w\s]', '', tag_str)
        tag_str = re.sub(r'\s+', ' ', tag_str)
        return tag_str.strip()
    
    def build_synonym_dict(self):
        synonym_groups = {
            'rock': ['rock', 'rocks', 'rock music'],
            'indie rock': ['indie rock', 'indierock'],
            'indie pop': ['indie pop', 'indiepop'],
            'electronic': ['electronic', 'electronica', 'electro', 'electronic music'],
            'alternative': ['alternative', 'alternative rock', 'alt rock', 'alt'],
            'pop': ['pop', 'pop music', 'pop rock', 'poprock'],
            'metal': ['metal', 'heavy metal', 'heavymetal'],
            'hip hop': ['hip hop', 'hiphop', 'hip-hop', 'rap'],
        }
        
        variant_to_canonical = {}
        for canonical, variants in synonym_groups.items():
            for variant in variants:
                variant_to_canonical[variant] = canonical
        return variant_to_canonical
    
    def apply_normalization(self, min_frequency=2):
        print("\nNormalizing tags...")
        tag_counts = self.user_taggedartists.groupby('tagID').size()
        synonym_dict = self.build_synonym_dict()

        normalized_tags = {}
        removed = 0

        for _, tag_row in self.tags.iterrows():
            tid = tag_row['tagID']
            tval = tag_row['tagValue']

            # skip infrequent tags
            if tid in tag_counts and tag_counts[tid] < min_frequency:
                removed += 1
                continue

            norm = self.normalize_tag_string(tval)
            if norm in synonym_dict:
                norm = synonym_dict[norm]

            normalized_tags[tid] = norm

        print(f"Kept {len(normalized_tags)} tags (removed {removed})")
        self.tag_mapping = normalized_tags
        return normalized_tags
    
    def get_clustered_tags_for_artists(self):
        filtered = self.user_taggedartists[
            self.user_taggedartists['tagID'].isin(self.tag_mapping.keys())
        ].copy()

        filtered['normalized_tag'] = filtered['tagID'].map(self.tag_mapping)

        artist_tags = filtered.groupby('artistID')['normalized_tag'].apply(
            lambda x: ' '.join(x.astype(str).tolist())
        ).reset_index()
        artist_tags.columns = ['artistID', 'tag_profile']

        print(f"Created tag profiles for {len(artist_tags)} artists")
        return artist_tags


# Neural collaborative filtering model
class MusicInteractionDataset(Dataset):
    def __init__(self, user_ids, artist_ids, ratings, negative_artist_ids=None):
        self.user_ids = torch.LongTensor(user_ids)
        self.artist_ids = torch.LongTensor(artist_ids)
        self.ratings = torch.FloatTensor(ratings)

        # negative sampling for BPR
        if negative_artist_ids is not None:
            self.negative_artist_ids = torch.LongTensor(negative_artist_ids)
            self.use_bpr = True
        else:
            self.negative_artist_ids = None
            self.use_bpr = False

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.use_bpr:
            return (self.user_ids[idx], self.artist_ids[idx],
                    self.negative_artist_ids[idx], self.ratings[idx])
        else:
            return self.user_ids[idx], self.artist_ids[idx], self.ratings[idx]


class BPRLoss(nn.Module):
    # Bayesian Personalized Ranking loss
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # positive items should rank higher than negative
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        return loss


class NeuralCF(nn.Module):
    def __init__(self, n_users, n_artists, embedding_dim=64, hidden_dims=[128, 64, 32],
                 dropout=0.2, use_gmf=True, use_mlp=True):
        super(NeuralCF, self).__init__()

        self.n_users = n_users
        self.n_artists = n_artists
        self.embedding_dim = embedding_dim
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp

        # GMF embeddings
        if use_gmf:
            self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
            self.gmf_artist_embedding = nn.Embedding(n_artists, embedding_dim)

        # MLP embeddings + layers
        if use_mlp:
            self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
            self.mlp_artist_embedding = nn.Embedding(n_artists, embedding_dim)

            mlp_layers = []
            input_dim = embedding_dim * 2
            for hidden_dim in hidden_dims:
                mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            self.mlp = nn.Sequential(*mlp_layers)
            mlp_output_dim = hidden_dims[-1] if hidden_dims else embedding_dim * 2

        # combine GMF and MLP
        if use_gmf and use_mlp:
            final_input_dim = embedding_dim + mlp_output_dim
        elif use_gmf:
            final_input_dim = embedding_dim
        else:
            final_input_dim = mlp_output_dim

        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, user_ids, artist_ids):
        gmf_output = None
        mlp_output = None
        
        if self.use_gmf:
            gmf_user_emb = self.gmf_user_embedding(user_ids)
            gmf_artist_emb = self.gmf_artist_embedding(artist_ids)
            gmf_output = gmf_user_emb * gmf_artist_emb
        
        if self.use_mlp:
            mlp_user_emb = self.mlp_user_embedding(user_ids)
            mlp_artist_emb = self.mlp_artist_embedding(artist_ids)
            mlp_input = torch.cat([mlp_user_emb, mlp_artist_emb], dim=1)
            mlp_output = self.mlp(mlp_input)
        
        if self.use_gmf and self.use_mlp:
            combined = torch.cat([gmf_output, mlp_output], dim=1)
        elif self.use_gmf:
            combined = gmf_output
        else:
            combined = mlp_output
        
        prediction = self.final_layer(combined)
        return prediction.squeeze()


class NeuralCFTrainer:

    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5, device='cpu', use_bpr=True):
        self.model = model.to(device)
        self.device = device
        self.use_bpr = use_bpr
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_bpr:
            self.criterion = BPRLoss()
            print("Using BPR Loss (Bayesian Personalized Ranking)")
        else:
            self.criterion = nn.BCELoss()
            print("Using BCE Loss (Binary Cross Entropy)")

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            if self.use_bpr:
                # BPR: (user, pos_item, neg_item, rating)
                users, pos_artists, neg_artists, ratings = batch
                users = users.to(self.device)
                pos_artists = pos_artists.to(self.device)
                neg_artists = neg_artists.to(self.device)

                self.optimizer.zero_grad()

                # Get predictions for positive and negative items
                pos_predictions = self.model(users, pos_artists)
                neg_predictions = self.model(users, neg_artists)

                # BPR loss
                loss = self.criterion(pos_predictions, neg_predictions)
            else:
                # BCE: (user, item, rating)
                users, artists, ratings = batch
                users = users.to(self.device)
                artists = artists.to(self.device)
                ratings = ratings.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(users, artists)
                loss = self.criterion(predictions, ratings)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.use_bpr:
                    # BPR validation
                    users, pos_artists, neg_artists, ratings = batch
                    users = users.to(self.device)
                    pos_artists = pos_artists.to(self.device)
                    neg_artists = neg_artists.to(self.device)

                    pos_predictions = self.model(users, pos_artists)
                    neg_predictions = self.model(users, neg_artists)
                    loss = self.criterion(pos_predictions, neg_predictions)
                else:
                    # BCE validation
                    users, artists, ratings = batch
                    users = users.to(self.device)
                    artists = artists.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = self.model(users, artists)
                    loss = self.criterion(predictions, ratings)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches
    
    def fit(self, train_loader, val_loader, n_epochs=10, early_stopping_patience=3):
        print(f"\n{'='*60}")
        print(f"Training Neural Collaborative Filtering (NCF)")
        print(f"{'='*60}")
        
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


# Main recommender class
class IntegratedRecommender:
    def __init__(self, data_dir='lastfm_data', embedding_dim=64):
        self.data_dir = Path(data_dir)
        self.embedding_dim = embedding_dim

        self.user_artists = None
        self.artists = None
        self.tags = None
        self.user_taggedartists = None
        
        self.ncf_model = None
        self.ncf_trainer = None
        self.user_id_map = {}  # For NCF
        self.artist_id_map = {}  # For NCF
        self.reverse_user_map = {}
        self.reverse_artist_map = {}
        
        self.artist_profiles = None
        self.artist_similarity = None  # TF-IDF cosine similarity
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self.artist_popularity = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Integrated system on device: {self.device}")
    
    def load_data(self, user_artists_df, artists_df, tags_df, user_taggedartists_df):
        self.user_artists = user_artists_df.copy()
        self.artists = artists_df.copy()
        self.tags = tags_df.copy()
        self.user_taggedartists = user_taggedartists_df.copy()

        print(f"Loaded {len(self.user_artists)} listening records")
        print(f"Loaded {len(self.artists)} artists")

    def build_content_model(self, use_clustering=True):
        print("\nBuilding content model...")
        
        if use_clustering:
            clusterer = TagClusterer(self.tags, self.user_taggedartists)
            clusterer.analyze_tag_quality()
            clusterer.apply_normalization(min_frequency=2)
            self.artist_profiles = clusterer.get_clustered_tags_for_artists()
            
            # merge with all artists (keep artists without tags)
            self.artist_profiles = self.artists[['id', 'name']].merge(
                self.artist_profiles,
                left_on='id',
                right_on='artistID',
                how='left'
            )
            self.artist_profiles['artistID'] = self.artist_profiles['id']

        # calculate popularity metrics
        self.artist_popularity = self.user_artists.groupby('artistID').agg({
            'weight': 'sum',
            'userID': 'nunique'
        }).rename(columns={'weight': 'total_plays', 'userID': 'unique_listeners'})

        self.artist_popularity['log_plays'] = np.log1p(self.artist_popularity['total_plays'])
        max_log = self.artist_popularity['log_plays'].max()
        min_log = self.artist_popularity['log_plays'].min()

        if max_log > min_log:
            self.artist_popularity['popularity_score'] = (
                (self.artist_popularity['log_plays'] - min_log) / (max_log - min_log)
            )
        else:
            self.artist_popularity['popularity_score'] = 0.5

        self.artist_popularity['popularity_score'] = self.artist_popularity['popularity_score'].fillna(0.0)
        self.artist_popularity['uniqueness_score'] = 1 - self.artist_popularity['popularity_score']
        self.artist_popularity['diversity_score'] = (
            self.artist_popularity['unique_listeners'] / self.artist_popularity['total_plays']
        ).fillna(0.5).clip(0, 1)
        self.artist_popularity['popularity_percentile'] = (
            self.artist_popularity['popularity_score'].rank(pct=True) * 100
        )
                
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.artist_profiles['tag_profile'].fillna('')
        )
        
        self.artist_similarity = cosine_similarity(self.tfidf_matrix)
        self.artist_similarity = np.nan_to_num(self.artist_similarity, nan=0.0)
        
        print(f"Content model ready - {len(self.artist_profiles)} artists")
    
    def generate_negative_samples(self, user_artists_df, n_neg_per_pos=4):
        """
        Generate negative samples for training with BPR loss

        For each positive user-artist interaction, sample N negative artists
        that the user has NOT interacted with.

        Args:
            user_artists_df: DataFrame with positive interactions
            n_neg_per_pos: Number of negative samples per positive sample

        Returns:
            DataFrame with columns: userID, artistID, negative_artistID, weight
        """
        print(f"\nGenerating negative samples ({n_neg_per_pos} per positive)...")

        all_artists = set(self.artist_id_map.keys())
        training_data = []

        # Group by user to generate negative samples efficiently
        for user_id, group in tqdm(user_artists_df.groupby('userID'), desc="Generating negatives"):
            if user_id not in self.user_id_map:
                continue

            # Get positive artists for this user
            positive_artists = set(group['artistID'].tolist())

            # Get candidate negative artists (all artists minus positive ones)
            negative_candidates = list(all_artists - positive_artists)

            if len(negative_candidates) == 0:
                continue

            # For each positive interaction, sample negative artists
            for _, row in group.iterrows():
                pos_artist = row['artistID']
                weight = row['weight']

                # Sample negative artists
                n_samples = min(n_neg_per_pos, len(negative_candidates))
                neg_artists = np.random.choice(negative_candidates, size=n_samples, replace=False)

                for neg_artist in neg_artists:
                    training_data.append({
                        'userID': user_id,
                        'artistID': pos_artist,  # Positive item
                        'negative_artistID': neg_artist,  # Negative item
                        'weight': weight
                    })

        result_df = pd.DataFrame(training_data)
        print(f"Generated {len(result_df):,} training samples")
        print(f"  Original interactions: {len(user_artists_df):,}")
        print(f"  Expansion ratio: {len(result_df) / len(user_artists_df):.1f}x")

        return result_df

    def build_ncf_model(self, use_bpr=True, n_neg_per_pos=4):
        print("\nBuilding NCF model...")

        # Create ID mappings
        unique_users = sorted(self.user_artists['userID'].unique())
        unique_artists = sorted(self.user_artists['artistID'].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.artist_id_map = {aid: idx for idx, aid in enumerate(unique_artists)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_artist_map = {idx: aid for aid, idx in self.artist_id_map.items()}
        
        print(f"NCF: {len(unique_users)} users, {len(unique_artists)} artists")

        self.use_bpr = use_bpr

        if use_bpr:
            # BPR mode - generate negative samples
            # First, generate negative samples for the full dataset
            df_with_negatives = self.generate_negative_samples(
                self.user_artists, n_neg_per_pos=n_neg_per_pos
            )

            # Map IDs
            df_with_negatives['user_idx'] = df_with_negatives['userID'].map(self.user_id_map)
            df_with_negatives['pos_artist_idx'] = df_with_negatives['artistID'].map(self.artist_id_map)
            df_with_negatives['neg_artist_idx'] = df_with_negatives['negative_artistID'].map(self.artist_id_map)

            # Normalize weights (for reference, not used in BPR loss directly)
            ratings = np.log1p(df_with_negatives['weight'].values)
            ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())

            user_indices = df_with_negatives['user_idx'].values
            pos_artist_indices = df_with_negatives['pos_artist_idx'].values
            neg_artist_indices = df_with_negatives['neg_artist_idx'].values

            # Split data
            (train_users, test_users, train_pos_artists, test_pos_artists,
             train_neg_artists, test_neg_artists, train_ratings, test_ratings) = train_test_split(
                user_indices, pos_artist_indices, neg_artist_indices, ratings,
                test_size=0.2, random_state=42
            )

            (train_users, val_users, train_pos_artists, val_pos_artists,
             train_neg_artists, val_neg_artists, train_ratings, val_ratings) = train_test_split(
                train_users, train_pos_artists, train_neg_artists, train_ratings,
                test_size=0.1, random_state=42
            )

            # Create datasets with negative samples
            train_dataset = MusicInteractionDataset(
                train_users, train_pos_artists, train_ratings, train_neg_artists
            )
            val_dataset = MusicInteractionDataset(
                val_users, val_pos_artists, val_ratings, val_neg_artists
            )

            print("Using BPR training with negative sampling")

        else:
            # BCE mode - no negative sampling
            df = self.user_artists.copy()
            df['user_idx'] = df['userID'].map(self.user_id_map)
            df['artist_idx'] = df['artistID'].map(self.artist_id_map)

            ratings = np.log1p(df['weight'].values)
            ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())

            user_indices = df['user_idx'].values
            artist_indices = df['artist_idx'].values

            # Split data
            train_users, test_users, train_artists, test_artists, train_ratings, test_ratings = train_test_split(
                user_indices, artist_indices, ratings,
                test_size=0.2, random_state=42
            )

            train_users, val_users, train_artists, val_artists, train_ratings, val_ratings = train_test_split(
                train_users, train_artists, train_ratings,
                test_size=0.1, random_state=42
            )

            # Create datasets without negative samples
            train_dataset = MusicInteractionDataset(train_users, train_artists, train_ratings)
            val_dataset = MusicInteractionDataset(val_users, val_artists, val_ratings)

            print("Using BCE training")

        self.train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        # Initialize NCF model
        self.ncf_model = NeuralCF(
            n_users=len(self.user_id_map),
            n_artists=len(self.artist_id_map),
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.2,
            use_gmf=True,
            use_mlp=True
        )
        
        n_params = sum(p.numel() for p in self.ncf_model.parameters())
        print(f"NCF model created ({n_params:,} parameters)")
    
    def train_ncf(self, n_epochs=10, use_bpr=True):
        self.ncf_trainer = NeuralCFTrainer(
            self.ncf_model,
            learning_rate=0.001,
            device=self.device,
            use_bpr=use_bpr
        )

        self.ncf_trainer.fit(
            self.train_loader,
            self.val_loader,
            n_epochs=n_epochs,
            early_stopping_patience=3
        )

        print("NCF training complete")
    
    def get_ncf_score(self, user_id, artist_id):
        if user_id not in self.user_id_map or artist_id not in self.artist_id_map:
            return 0.0

        user_idx = self.user_id_map[user_id]
        artist_idx = self.artist_id_map[artist_id]

        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            artist_tensor = torch.LongTensor([artist_idx]).to(self.device)

            score = self.ncf_model(user_tensor, artist_tensor)
            return float(score.cpu().item())

    def get_recommendations(self, user_id, n_recommendations=20,
                            ncf_weight=0.5, content_weight=0.5,
                            niche_slider=0.5, uniqueness_slider=0.5):
        """
        Get recommendations using TWO methods:
        1. NCF (collaborative filtering)
        2. TF-IDF (content-based)

        Plus niche slider logic on top
        """
        
        user_history = self.user_artists[self.user_artists['userID'] == user_id]
        if len(user_history) == 0:
            return []
        
        listened_artists = set(user_history['artistID'].tolist())

        # 1. Get NCF scores
        ncf_scores = {}
        if self.ncf_model is not None and user_id in self.user_id_map:
            for artist_id in self.artist_id_map.keys():
                if artist_id not in listened_artists:
                    ncf_scores[artist_id] = self.get_ncf_score(user_id, artist_id)

        # 2. Get content similarity scores
        top_artists = user_history.nlargest(5, 'weight')['artistID'].tolist()

        content_scores = defaultdict(float)
        for seed_artist in top_artists:
            if seed_artist not in self.artist_profiles['artistID'].values:
                continue

            artist_idx = self.artist_profiles[
                self.artist_profiles['artistID'] == seed_artist
            ].index[0]

            sim_scores = self.artist_similarity[artist_idx]

            for idx, score in enumerate(sim_scores):
                candidate_id = self.artist_profiles.iloc[idx]['artistID']
                if candidate_id not in listened_artists and score >= 0.05:
                    content_scores[candidate_id] += score
        
        # Normalize all scores
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {}
            values = list(scores_dict.values())
            max_val = max(values)
            min_val = min(values)
            if max_val > min_val:
                return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
            return {k: 0.5 for k in scores_dict.keys()}

        ncf_scores = normalize_scores(ncf_scores)
        content_scores = normalize_scores(dict(content_scores))

        # 3. Combine NCF and content scores
        all_candidates = set(ncf_scores.keys()) | set(content_scores.keys())
        
        candidate_list = []
        for artist_id in all_candidates:
            if artist_id not in self.artist_profiles['artistID'].values:
                continue

            ncf_score = ncf_scores.get(artist_id, 0.0)
            cont_score = content_scores.get(artist_id, 0.0)

            # Weighted combination of BOTH
            hybrid_score = (ncf_weight * ncf_score + content_weight * cont_score)
            
            # Get artist info
            artist_info = self.artist_profiles[
                self.artist_profiles['artistID'] == artist_id
            ].iloc[0]

            # 4. Apply niche boost
            if artist_id in self.artist_popularity.index:
                pop_data = self.artist_popularity.loc[artist_id]
                popularity = pop_data['popularity_score']
                uniqueness = pop_data['uniqueness_score']
                diversity = pop_data['diversity_score']
                popularity_percentile = pop_data['popularity_percentile']
            else:
                popularity = 0.5
                uniqueness = 0.5
                diversity = 0.5
                popularity_percentile = 50.0

            # NICHE BOOST LOGIC
            niche_boost = 1.0
            uniqueness_boost = 1.0
            
            if niche_slider > 0:
                if popularity > 0.5:
                    popularity_excess = (popularity - 0.5) * 2
                    penalty_strength = niche_slider * (popularity_excess ** 2)
                    niche_boost = 1.0 - (penalty_strength * 0.99)
                else:
                    unpopularity = (0.5 - popularity) * 2
                    boost_strength = niche_slider * (unpopularity ** 1.5)
                    niche_boost = 1.0 + (boost_strength * 7.0)

            if uniqueness_slider > 0:
                uniqueness_boost = 1.0 + (uniqueness_slider * diversity * 1.5)

            niche_boost = max(0.01, min(8.0, niche_boost))
            uniqueness_boost = max(1.0, min(2.5, uniqueness_boost))
            
            final_score = hybrid_score * niche_boost * uniqueness_boost
            
            candidate_list.append({
                'artistID': artist_id,
                'name': artist_info['name'],
                'final_score': float(final_score),
                'hybrid_score': float(hybrid_score),
                'ncf_score': float(ncf_score),
                'content_score': float(cont_score),
                'popularity_percentile': float(popularity_percentile),
                'niche_boost': float(niche_boost),
            })
        
        candidate_list = sorted(candidate_list, key=lambda x: x['final_score'], reverse=True)
        return candidate_list[:n_recommendations]
    
    def train(self, use_tag_clustering=True, ncf_epochs=10, use_bpr=True, n_neg_per_pos=4):
        # Train both NCF and content models
        # use_tag_clustering: whether to use tag clustering
        # ncf_epochs: number of training epochs
        # use_bpr: use BPR loss (otherwise BCE)
        print("\nTraining integrated system...")
        if use_bpr:
            print("Using BPR loss with negative sampling")
        else:
            print("Using BCE loss")

        # 1. Build content model
        self.build_content_model(use_clustering=use_tag_clustering)

        # 2. Build and train NCF
        self.build_ncf_model(use_bpr=use_bpr, n_neg_per_pos=n_neg_per_pos)
        self.train_ncf(n_epochs=ncf_epochs, use_bpr=use_bpr)

        print("\nTraining complete!")
        print("  NCF:", "BPR loss" if use_bpr else "BCE loss")
        print("  Content: TF-IDF")


# Example usage

if __name__ == "__main__":
    # Load data
    print("Loading Last.fm data...")
    base_path = Path('lastfm_data')
    
    user_artists = pd.read_csv(base_path / "user_artists.dat", sep='\t', encoding='latin-1')
    
    try:
        artists = pd.read_csv(base_path / "artists.dat", sep='\t', encoding='utf-8', on_bad_lines='skip')
    except:
        artists = pd.read_csv(base_path / "artists.dat", sep='\t', encoding='latin-1', on_bad_lines='skip')
    
    tags = pd.read_csv(base_path / "tags.dat", sep='\t', encoding='latin-1')
    user_taggedartists = pd.read_csv(base_path / "user_taggedartists.dat", sep='\t', encoding='latin-1')
    
    print("Data loaded")

    # Create integrated recommender
    recommender = IntegratedRecommender(embedding_dim=64)

    # Load data
    recommender.load_data(user_artists, artists, tags, user_taggedartists)

    # Train BOTH models with BPR loss
    recommender.train(
        use_tag_clustering=True,
        ncf_epochs=5,  # Quick demo
        use_bpr=True,  # Use BPR loss 
        n_neg_per_pos=4  # 4 negative samples per positive
    )
    
    # Test recommendations
    user_id = 7
    
    print(f"\nTesting recommendations for user {user_id}")

    # Show user history
    user_history = user_artists[user_artists['userID'] == user_id]
    print(f"\nUser {user_id}'s top artists:")
    top_history = user_history.nlargest(5, 'weight')
    for _, row in top_history.iterrows():
        artist_name = artists[artists['id'] == row['artistID']]['name'].values
        if len(artist_name) > 0:
            print(f"  ♫ {artist_name[0]} ({row['weight']} plays)")

    # Get recommendations with BOTH models
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS (NCF=50%, Content=50%)")
    print(f"{'='*80}")

    recs = recommender.get_recommendations(
        user_id,
        n_recommendations=20,
        ncf_weight=0.5,
        content_weight=0.5,
        niche_slider=0.5,
        uniqueness_slider=0.5
    )

    print(f"\n{'Rank':<6}{'Artist':<35}{'Final':<8}{'NCF':<8}{'Cont':<8}{'Pop%':<7}")
    print("-" * 75)

    for i, rec in enumerate(recs, 1):
        name = rec['name'][:32] + '...' if len(rec['name']) > 32 else rec['name']
        print(f"{i:<6}{name:<35}{rec['final_score']:<8.3f}{rec['ncf_score']:<8.3f}"
              f"{rec['content_score']:<8.3f}{rec['popularity_percentile']:<7.0f}")

    print("\n" + "="*80)
    print("System ready!")
    print("="*80)
    print("\nYou can now adjust weights:")
    print("  recs = recommender.get_recommendations(")
    print("      user_id=7,")
    print("      ncf_weight=0.5,     # Collaborative filtering")
    print("      content_weight=0.5, # TF-IDF")
    print("      niche_slider=0.5    # Niche control")
    print("  )")