import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from gensim.models import Word2Vec
import torchtext
from torchtext.vocab import GloVe
from transformers import BertModel, BertTokenizerFast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from umap import UMAP
import warnings
import logging
import spacy  # Import for spaCy

torchtext.disable_torchtext_deprecation_warning()
warnings.filterwarnings("ignore")

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TextPreprocessor:
    def __init__(
        self,
        target_column: str = 'text',
        include_stopwords: bool = True,
        remove_ats: bool = True,
        word_limit: int = 100,
        tokenizer: Optional[Any] = None
    ):
        self.target_column = target_column
        self.include_stopwords = include_stopwords
        self.remove_ats = remove_ats
        self.word_limit = word_limit
        self.tokenizer = tokenizer if tokenizer else self.spacy_tokenizer

        if include_stopwords:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                self.stop_words = self.nlp.Defaults.stop_words
            except OSError:
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                    self.stop_words = self.nlp.Defaults.stop_words
                except Exception as e:
                    raise OSError(f"Failed to install the spaCy model 'en_core_web_sm': {e}")
        else:
            self.stop_words = set()

        self.re_pattern = re.compile(r'[^\w\s]')
        self.at_pattern = re.compile(r'@\S+')

    def spacy_tokenizer(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.text for token in doc]

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in df.columns:
            raise ValueError(f"The target column '{self.target_column}' does not exist.")

        df = df.copy()
        df[self.target_column] = df[self.target_column].astype(str).str.lower()

        if self.remove_ats:
            df[self.target_column] = df[self.target_column].str.replace(self.at_pattern, '', regex=True)

        df[self.target_column] = df[self.target_column].str.replace(self.re_pattern, '', regex=True)
        df['tokenized_text'] = df[self.target_column].apply(self.tokenizer)

        if self.include_stopwords:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if word not in self.stop_words and len(word) <= self.word_limit]
            )
        else:
            df['tokenized_text'] = df['tokenized_text'].apply(
                lambda tokens: [word for word in tokens if len(word) <= self.word_limit]
            )

        return df


class EmbeddingCreator:
    def __init__(
        self,
        embedding_method: str = "bert",
        embedding_dim: int = 768,
        glove_cache_path: Optional[str] = None,
        word2vec_model_path: Optional[str] = None,
        bert_model_name: str = "bert-base-uncased",
        bert_cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        self.embedding_method = embedding_method.lower()
        self.embedding_dim = embedding_dim
        self.glove = None
        self.word2vec_model = None
        self.bert_model = None
        self.tokenizer = None

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if self.embedding_method == "glove":
            self._load_glove(glove_cache_path)
        elif self.embedding_method == "word2vec":
            if word2vec_model_path:
                self._load_word2vec(word2vec_model_path)
            else:
                self.word2vec_model = None  # To be trained later
        elif self.embedding_method == "bert":
            self._load_bert(bert_model_name, bert_cache_dir)
        else:
            raise ValueError("Unsupported embedding method. Choose from 'glove', 'word2vec', or 'bert'.")

    def _load_glove(self, glove_cache_path: str):
        if not glove_cache_path:
            raise ValueError("glove_cache_path must be provided for GloVe embeddings.")
        if not os.path.exists(glove_cache_path):
            raise FileNotFoundError(f"GloVe cache path '{glove_cache_path}' does not exist.")
        self.glove = GloVe(name="6B", dim=self.embedding_dim, cache=glove_cache_path)

    def _load_word2vec(self, word2vec_model_path: str):
        if not word2vec_model_path or not os.path.exists(word2vec_model_path):
            raise ValueError("A valid word2vec_model_path must be provided for Word2Vec embeddings.")
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        if self.word2vec_model.vector_size != self.embedding_dim:
            raise ValueError(f"Word2Vec model dimension ({self.word2vec_model.vector_size}) "
                             f"does not match embedding_dim ({self.embedding_dim}).")

    def _load_bert(self, bert_model_name: str, bert_cache_dir: Optional[str]):
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=bert_cache_dir)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.embedding_dim = self.bert_model.config.hidden_size

    def train_word2vec(self, sentences: List[List[str]], vector_size: int = 300,
                       window: int = 5, min_count: int = 1, workers: int = 4):
        if self.embedding_method != "word2vec":
            raise ValueError("train_word2vec can only be called for 'word2vec' embedding method.")
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=42
        )

    def get_embedding(self, tokens: List[str]) -> np.ndarray:
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_average_embedding(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_embedding(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def get_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        if self.embedding_method in ["glove", "word2vec"]:
            if self.embedding_method == "word2vec" and self.word2vec_model is None:
                raise ValueError("Word2Vec model is not trained. Please train the model before getting embeddings.")
            return self._get_individual_embeddings(tokens)
        elif self.embedding_method == "bert":
            return self._get_bert_word_embeddings(tokens)
        else:
            raise ValueError("Unsupported embedding method.")

    def _get_average_embedding(self, tokens: List[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            if self.embedding_method == "glove" and token in self.glove.stoi:
                embeddings.append(self.glove[token].numpy())
            elif self.embedding_method == "word2vec" and token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                continue
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)

    def _get_individual_embeddings(self, tokens: List[str]) -> np.ndarray:
        embeddings = []
        for token in tokens:
            if self.embedding_method == "glove" and token in self.glove.stoi:
                embeddings.append(self.glove[token].numpy())
            elif self.embedding_method == "word2vec" and token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                embeddings.append(np.zeros(self.embedding_dim))
        return np.array(embeddings)

    def _get_bert_embedding(self, tokens: List[str]) -> np.ndarray:
        text = ' '.join(tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    def _get_bert_word_embeddings(self, tokens: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state.squeeze(0)
            word_ids = inputs.word_ids(batch_index=0)
            if word_ids is None:
                raise ValueError("word_ids() returned None. Ensure you are using a fast tokenizer.")
            word_embeddings = {}
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id not in word_embeddings:
                        word_embeddings[word_id] = []
                    word_embeddings[word_id].append(last_hidden_state[idx].cpu().numpy())
            averaged_embeddings = []
            for wid in sorted(word_embeddings.keys()):
                arr = np.array(word_embeddings[wid])
                averaged_embeddings.append(arr.mean(axis=0))
        return np.array(averaged_embeddings)


class FeatureAggregatorSimple(nn.Module):
    def __init__(
        self,
        sentence_dim: int,
        categorical_columns: List[str],
        categorical_dims: Dict[str, int],
        categorical_embed_dim: int
    ):
        super(FeatureAggregatorSimple, self).__init__()
        self.categorical_columns = categorical_columns
        self.categorical_dims = categorical_dims
        self.categorical_embed_dim = categorical_embed_dim

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=dim, embedding_dim=categorical_embed_dim)
            for col, dim in categorical_dims.items()
        })

        self.sentence_projection = nn.Linear(sentence_dim, sentence_dim)
        self.categorical_projection = nn.Linear(categorical_embed_dim, sentence_dim)
        self.weights = {col: 1.0 for col in categorical_columns}

    def set_categorical_weights(self, weights: Dict[str, float]):
        self.weights = weights

    def forward(self, sentence_embeddings: torch.Tensor, categorical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        aggregated_features = self.sentence_projection(sentence_embeddings)

        for col in self.categorical_columns:
            if col in categorical_data:
                embedded = self.embeddings[col](categorical_data[col])
                embedded = self.categorical_projection(embedded)
                weight = self.weights.get(col, 1.0)
                aggregated_features += weight * embedded

        aggregated_features = torch.relu(aggregated_features)
        return aggregated_features


class FeatureSpaceCreator:
    def __init__(self, config: Dict[str, Any], device: str = "cuda", log_file: str = "logs/feature_space_creator.log"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.features = config.get("features", [])
        self.multi_graph_settings = config.get("multi_graph_settings", {})

        self.text_features = []
        self.numeric_features = []

        self.logger = self._setup_logger(log_file=log_file)
        self._parse_config()

        self.embedding_creators = {}
        for feature in self.text_features:
            method = feature.get("embedding_method", "bert").lower()
            embedding_dim = feature.get("embedding_dim", None)
            additional_params = feature.get("additional_params", {})

            try:
                self.embedding_creators[feature["column_name"]] = EmbeddingCreator(
                    embedding_method=method,
                    embedding_dim=embedding_dim,
                    glove_cache_path=additional_params.get("glove_cache_path"),
                    word2vec_model_path=additional_params.get("word2vec_model_path"),
                    bert_model_name=additional_params.get("bert_model_name", "bert-base-uncased"),
                    bert_cache_dir=additional_params.get("bert_cache_dir"),
                    device=self.device
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize EmbeddingCreator for '{feature['column_name']}': {e}")
                raise e

        self.scalers = {}
        for feature in self.numeric_features:
            processing = feature.get("processing", "none").lower()
            if processing == "standardize":
                self.scalers[feature["column_name"]] = StandardScaler()
            elif processing == "normalize":
                self.scalers[feature["column_name"]] = MinMaxScaler()

        self.projection_layers = {}
        for feature in self.numeric_features:
            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                projection = nn.Linear(1, target_dim).to(self.device)
                projection.eval()
                self.projection_layers[feature["column_name"]] = projection

        self.text_preprocessor = TextPreprocessor(
            target_column=None,
            include_stopwords=True,
            remove_ats=True,
            word_limit=100,
            tokenizer=None
        )

    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("FeatureSpaceCreator")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _parse_config(self):
        for feature in self.features:
            f_type = feature.get("type", "").lower()
            if f_type == "text":
                self.text_features.append(feature)
            elif f_type == "numeric":
                self.numeric_features.append(feature)
            else:
                raise ValueError(f"Unsupported feature type: '{f_type}' in feature '{feature.get('column_name')}'.")

    def process(self, dataframe: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(dataframe, str):
            if not os.path.exists(dataframe):
                raise FileNotFoundError(f"CSV file not found at path: {dataframe}")
            df = pd.read_csv(dataframe)
            self.logger.info(f"Loaded data from '{dataframe}'.")
        elif isinstance(dataframe, pd.DataFrame):
            df = dataframe.copy()
            self.logger.info("Loaded data from pandas DataFrame.")
        else:
            raise TypeError("dataframe must be a file path (str) or a pandas DataFrame.")

        feature_space = pd.DataFrame(index=df.index)
        self.logger.info("Initialized feature space DataFrame.")

        # Process text features
        for feature in self.text_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Text column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in text column '{col}'. Filling with empty strings.")
                df[col] = df[col].fillna("")

            if self.text_preprocessor.target_column != col:
                self.text_preprocessor.target_column = col
            processed_df = self.text_preprocessor.clean_text(df)
            tokens = processed_df["tokenized_text"].tolist()

            if feature["embedding_method"].lower() == "word2vec":
                word2vec_model_path = feature.get("additional_params", {}).get("word2vec_model_path", None)
                if not word2vec_model_path:
                    self.logger.info(f"Training Word2Vec model for '{col}' as no model path was provided.")
                    self.embedding_creators[col].train_word2vec(sentences=tokens)
                    self.logger.info(f"Word2Vec model trained for '{col}'.")
                else:
                    pass

            embeddings = []
            for token_list in tokens:
                embedding = self.embedding_creators[col].get_embedding(token_list)
                embeddings.append(embedding)

            embeddings_array = np.vstack(embeddings)
            self.logger.info(f"Generated embeddings for text column '{col}' with shape {embeddings_array.shape}.")

            dim_reduction_config = feature.get("dim_reduction", {})
            method = dim_reduction_config.get("method", "none").lower()
            target_dim = dim_reduction_config.get("target_dim", embeddings_array.shape[1])

            if method in ["pca", "umap"] and target_dim < embeddings_array.shape[1]:
                self.logger.info(f"Applying '{method}' to text feature '{col}' to reduce dimensions to {target_dim}.")
                if method == "pca":
                    reducer = PCA(n_components=target_dim, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)
                elif method == "umap":
                    n_neighbors = dim_reduction_config.get("n_neighbors", 15)
                    min_dist = dim_reduction_config.get("min_dist", 0.1)
                    reducer = UMAP(n_components=target_dim, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    reduced_embeddings = reducer.fit_transform(embeddings_array)

                embeddings_array = reduced_embeddings
                self.logger.info(f"Dimensionality reduction '{method}' applied to '{col}'. "
                                 f"New shape: {embeddings_array.shape}.")

            feature_space[f"{col}_embedding"] = list(embeddings_array)

        # Process numeric features
        for feature in self.numeric_features:
            col = feature["column_name"]
            if col not in df.columns:
                raise ValueError(f"Numeric column '{col}' not found in the DataFrame.")

            if df[col].isnull().any():
                self.logger.warning(f"Missing values found in numeric column '{col}'. Filling with column mean.")
                df[col] = df[col].fillna(df[col].mean())

            data_type = feature.get("data_type", "float").lower()
            if data_type not in ["int", "float"]:
                raise ValueError(f"Unsupported data_type '{data_type}' for numeric column '{col}'.")

            df[col] = df[col].astype(float) if data_type == "float" else df[col].astype(int)

            processing = feature.get("processing", "none").lower()
            if processing in ["standardize", "normalize"]:
                scaler = self.scalers[col]
                df_scaled = scaler.fit_transform(df[[col]])
                feature_vector = df_scaled.flatten()
                self.logger.info(f"Applied '{processing}' to numeric column '{col}'.")
            else:
                feature_vector = df[col].values.astype(float)
                self.logger.info(f"No scaling applied to numeric column '{col}'.")

            projection_config = feature.get("projection", {})
            method = projection_config.get("method", "none").lower()
            target_dim = projection_config.get("target_dim", 1)

            if method == "linear" and target_dim > 1:
                self.logger.info(f"Applying '{method}' projection to numeric feature '{col}' to increase dimensions to {target_dim}.")
                projection_layer = self.projection_layers[col]
                with torch.no_grad():
                    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(1).to(self.device)
                    projected_tensor = projection_layer(feature_tensor)
                    projected_features = projected_tensor.cpu().numpy()
                feature_space[f"{col}_feature"] = list(projected_features)
                self.logger.info(f"Projection '{method}' applied to '{col}'. New shape: {projected_features.shape}.")
            else:
                feature_space[f"{col}_feature"] = feature_vector
                self.logger.info(f"Added numeric feature '{col}' with shape {feature_vector.shape}.")

        self.logger.info("Feature space creation completed.")
        return feature_space

    def aggregate_features(
        self,
        feature_space: pd.DataFrame,
        categorical_columns: List[str],
        categorical_dims: Dict[str, int],
        sentence_dim: int = 768
    ) -> torch.Tensor:
        sentence_embedding_cols = [col for col in feature_space.columns if col.endswith("_embedding")]
        if not sentence_embedding_cols:
            raise ValueError("No sentence embedding columns found in feature_space.")
        elif len(sentence_embedding_cols) > 1:
            self.logger.warning(f"Multiple sentence embedding columns found: {sentence_embedding_cols}. Using the first one.")

        sentence_col = sentence_embedding_cols[0]
        sentence_embeddings = torch.tensor(feature_space[sentence_col].tolist(), dtype=torch.float32).to(self.device)

        categorical_data = {}
        for col in categorical_columns:
            if col not in feature_space.columns:
                raise ValueError(f"Categorical column '{col}' not found in feature_space.")
            cat_values = feature_space[col].values
            max_index = cat_values.max()
            if max_index >= categorical_dims[col]:
                raise ValueError(f"Categorical column '{col}' has index {max_index} "
                                 f"which exceeds its dimension {categorical_dims[col]}.")
            categorical_data[col] = torch.tensor(cat_values, dtype=torch.long).to(sentence_embeddings.device)

        aggregator = FeatureAggregatorSimple(
            sentence_dim=sentence_dim,
            categorical_columns=categorical_columns,
            categorical_dims=categorical_dims,
            categorical_embed_dim=sentence_dim
        ).to(sentence_embeddings.device)

        weights_dict = {col: 1.0 for col in categorical_columns}
        aggregator.set_categorical_weights(weights_dict)

        aggregator.eval()

        with torch.no_grad():
            final_features = aggregator(sentence_embeddings, categorical_data)

        self.logger.info("Aggregated features successfully.")
        return final_features
