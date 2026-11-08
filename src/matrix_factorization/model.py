import numpy as np

class MatrixFactorization:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors=20,
        learning_rate=0.01,
        reg=0.05,
        n_epochs=20,
        random_state=42
    ):
        # Model configuration
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs

        # Random generator
        self.rng = np.random.default_rng(random_state)

        # Latent factors
        self.user_factors = self.rng.normal(
            0, 0.1, size=(n_users, n_factors)
        )

        self.item_factors = self.rng.normal(
            0, 0.1, size=(n_items, n_factors)
        )

        # User and item biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Global mean rating
        self.global_mean = 0.0

        # Training history
        self.history = []

    def predict(self, user_idx, item_idx):
        prediction = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(
                self.user_factors[user_idx],
                self.item_factors[item_idx]
            )
        )

        return prediction

    def compute_rmse(
        self,
        user_indices,
        item_indices,
        ratings
    ):
        predictions = (
            self.global_mean
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
            + np.sum(
                self.user_factors[user_indices]
                * self.item_factors[item_indices],
                axis=1
            )
        )

        return np.sqrt(
            np.mean((ratings - predictions) ** 2)
        )
    
    def fit(
        self,
        user_indices,
        item_indices,
        ratings,
        val_user_indices=None,
        val_item_indices=None,
        val_ratings=None
    ):
        # Initialize global mean using training ratings only
        self.global_mean = float(np.mean(ratings))

        n_samples = len(ratings)

        self.history = []
        self.val_history = []

        print("========== MF TRAINING ==========")
        print("Users:", self.n_users)
        print("Items:", self.n_items)
        print("Factors:", self.n_factors)
        print("Learning rate:", self.learning_rate)
        print("Regularization:", self.reg)
        print("Epochs:", self.n_epochs)

        for epoch in range(self.n_epochs):

            # Shuffle interactions each epoch
            order = self.rng.permutation(n_samples)

            for idx in order:
                u = user_indices[idx]
                i = item_indices[idx]
                actual = ratings[idx]

                # Predict rating
                predicted = self.predict(u, i)

                # Prediction error
                error = actual - predicted

                # Copy old latent vectors
                old_user = self.user_factors[u].copy()
                old_item = self.item_factors[i].copy()

                # Update biases
                # Update user bias
                self.user_bias[u] += self.learning_rate * (
                    error - self.reg * self.user_bias[u]
                )

                # self.item_bias[i] += self.learning_rate * (
                #     error * old_user - self.reg * old_item
                # )

                # Update item bias
                self.item_bias[i] += self.learning_rate * (
                    error - self.reg * self.item_bias[i]
                )

                # Update latent factors
                self.user_factors[u] += self.learning_rate * (
                    error * old_item - self.reg * old_user
                )

                self.item_factors[i] += self.learning_rate * (
                    error * old_user - self.reg * old_item
                )

            # Calculate training RMSE
            train_rmse = self.compute_rmse(
                user_indices,
                item_indices,
                ratings
            )

            self.history.append(train_rmse)

            if val_ratings is not None:
                val_rmse = self.compute_rmse(
                    val_user_indices,
                    val_item_indices,
                    val_ratings
                )

                self.val_history.append(val_rmse)

                print(
                    f"Epoch {epoch + 1:02d}/{self.n_epochs} "
                    f"- Train RMSE: {train_rmse:.4f} "
                    f"- Validation RMSE: {val_rmse:.4f}"
                )

            else:
                print(
                    f"Epoch {epoch + 1:02d}/{self.n_epochs} "
                    f"- Train RMSE: {train_rmse:.4f}"
                )

        print("\nMF-Step2 Completed.")

        return self
