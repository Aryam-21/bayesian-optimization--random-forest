import time
import logging
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


logger = logging.getLogger(__name__)


class RandomForestRandomSearch:
    """
    Randomized hyperparameter search for a Random Forest classifier.
    """

    def __init__(
        self,
        X_train,
        y_train,
        search_space: Dict[str, Any],
        n_iter: int = 20,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.search_space = search_space
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.search = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.runtime = None

    def _validate_inputs(self) -> None:
        """Validate search configuration before running."""

        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data cannot be None.")

        if not self.search_space:
            raise ValueError("Search space cannot be empty.")

        if self.n_iter <= 0:
            raise ValueError("n_iter must be greater than 0.")

        if self.cv < 2:
            raise ValueError("cv must be at least 2.")

    def _create_model(self) -> RandomForestClassifier:
        """Create the base Random Forest model."""

        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def run(self) -> Dict[str, Any]:
        """
        Run RandomizedSearchCV.

        Returns:
            Dictionary containing best parameters, CV score,
            runtime, and fitted model.
        """

        try:
            self._validate_inputs()

            logger.info("Starting Random Forest Random Search...")

            model = self._create_model()

            self.search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.search_space,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                return_train_score=False,
            )

            start_time = time.time()

            self.search.fit(self.X_train, self.y_train)

            self.runtime = time.time() - start_time

            self.best_model = self.search.best_estimator_
            self.best_params = self.search.best_params_
            self.best_score = self.search.best_score_

            logger.info("Random Search completed successfully.")
            logger.info("Best CV score: %.4f", self.best_score)
            logger.info("Runtime: %.2f seconds", self.runtime)

            return {
                "best_params": self.best_params,
                "best_cv_score": self.best_score,
                "runtime": self.runtime,
                "best_model": self.best_model,
                "search": self.search,
            }

        except ValueError as error:
            logger.error("Invalid Random Search configuration: %s", error)
            raise

        except Exception as error:
            logger.exception("Random Search failed: %s", error)
            raise RuntimeError(
                "Random Forest Random Search failed."
            ) from error

    def get_cv_results(self):
        """Return the detailed cross-validation results."""

        if self.search is None:
            raise RuntimeError(
                "Search has not been executed. Call run() first."
            )

        return self.search.cv_results_

