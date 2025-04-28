import logging
import time

import numpy as np
import pandas as pd
import torch

from .._explanation import Explanation
from ..utils import safe_isinstance
from ..utils._exceptions import ExplainerError, DimensionError
from ..utils._legacy import convert_to_instance, convert_to_model, match_instance_to_data, convert_to_data, \
    match_model_to_data
from ._explainer import Explainer

log = logging.getLogger("shap")


class TimeExplainer(Explainer):
    """Computes SHAP values based on time-series data.
    
    TimeExplainer is designed to explain models that work with time-series data.
    It extends the SHAP framework to handle temporal dependencies and patterns.
    
    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features x # time steps) and
        computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).
        
    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. For time-series data,
        this should include temporal information.
        
    feature_names : list
        The names of the features in the background dataset.
        
    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Default is "identity" (a no-op).
    """
    
    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):
        # Initialize the explainer
        # super().__init__(model, masker=data, link=link, feature_names=feature_names, **kwargs)
        
        # Store any additional parameters specific to time-series data
        self.time_steps = kwargs.get("time_steps", None)
        self.time_window = kwargs.get("time_window", None)

        # TODO - what do we do with this?
        self.keep_index = kwargs.get("keep_index", False)



        # Convert incoming inputs to standardized objects
        # print(type(model))
        self.model = model
        # self.model = convert_to_model(model)

        # self.model = convert_to_model(model, keep_index=self.keep_index)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        # model_null = match_model_to_data(self.model, self.data)
        
        # Set up data properties
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
            
        # Log initialization
        log.info("Initialized TimeExplainer")
        
    def __call__(self, X, **kwargs):
        """Explains the prediction for X.
        
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            The input to explain, typically a time-series.
            
        Returns
        -------
        shap.Explanation
            An explanation object with the SHAP values and related information.
        """
        start_time = time.time()
        
        # Extract feature names if available
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = self.feature_names
            
        # Calculate SHAP values
        shap_values = self.shap_values(X, **kwargs)
        
        # Create and return an Explanation object
        return Explanation(
            shap_values,
            base_values=getattr(self, "expected_value", 0),
            data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=feature_names,
            compute_time=time.time() - start_time
        )

    def shap_values(self, X, **kwargs):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array
            A matrix of samples (#batch_size x # time_steps x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "aic", "bic", or float
            The l1 regularization to use for feature selection. The estimation
            procedure is based on a debiased lasso.

            * "num_features(int)" selects a fixed number of top features.
            * "aic" and "bic" options use the AIC and BIC rules for regularization.
            * Passing a float directly sets the "alpha" parameter of the
              ``sklearn.linear_model.Lasso`` model used for feature selection.
            * "auto" (deprecated): uses "aic" when less than
              20% of the possible sample space is enumerated, otherwise it uses
              no regularization.

            .. versionchanged:: 0.47.0
                The default value changed from ``"auto"`` to ``"num_features(10)"``.

        silent: bool
            If True, hide tqdm progress bar. Default False.

        gc_collect : bool
           Run garbage collection after each explanation round. Sometime needed for memory intensive explanations (default False).

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as the ``expected_value``
            attribute of the explainer).

            The type and shape of the return value depends on the number of model inputs and outputs:

            * one input, one output: array of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: array of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs: list of arrays of corresponding shape above.

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs and one input changed from list to np.ndarray.

        """
        log.info("Calculating SHAP values...")

        # TODO - do we need this?
        # convert dataframes
        # if isinstance(X, pd.Series):
        #     X = X.values
        # elif isinstance(X, pd.DataFrame):
        #     if self.keep_index:
        #         index_value = X.index.values
        #         index_name = X.index.name
        #         column_name = list(X.columns)
        #     X = X.values
        #
        # x_type = str(type(X))
        # arr_type = "'numpy.ndarray'>"
        #
        # # if sparse, convert to lil for performance
        # if scipy.sparse.issparse(X) and not scipy.sparse.isspmatrix_lil(X):
        #     X = X.tolil()
        # assert x_type.endswith(arr_type) or scipy.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type

        # single instance
        if len(X.shape) == 2:
            data = X.reshape(1, *X.shape)
            # if self.keep_index:
            #     data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            out = np.zeros(s)
            out[:] = explanation
            return out

        # explain the whole dataset
        elif len(X.shape) == 3:
            emsg = "Not yet implemented for matrix of samples, only single instances!"
            raise ExplainerError(emsg)
        else:
            emsg = "Instance must have 2 or 3 dimensions!"
            raise DimensionError(emsg)
    
    def explain(self, incoming_instance, **kwargs):
        """Explains a single row and returns the explanation data.
        
        This method implements the abstract method from the Explainer base class.
        
        Returns
        -------
        dict
            A dictionary containing the explanation data.
        """
        # log.info(incoming_instance)
        log.info(type(self.model))
        # log.info(incoming_instance)
        with torch.no_grad():
            out = self.model(incoming_instance)
        print(out[0])

        # log.info(out)
        
        return out[0].detach().numpy()
    
    @staticmethod
    def supports_model_with_masker(model, masker):
        """Determines if this explainer can handle the given model and masker.
        
        Parameters
        ----------
        model : object
            The model to explain.
        masker : object
            The masker to use for masking features.
            
        Returns
        -------
        bool
            Whether this explainer supports the given model and masker.
        """
        # This is a placeholder implementation
        # In a real implementation, this would check if the model and masker are compatible
        return True