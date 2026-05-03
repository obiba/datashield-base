import logging
from typing import Any

from datashield import DSSession

logger = logging.getLogger(__name__)


class StatsClient:
    """Service for performing statistical operations on DataSHIELD sessions.

    This client provides methods for retrieving various statistics and metadata about symbols in remote R sessions
    of a DataSHIELD session, such as classes, lengths, levels, frequencies, quantiles, means, names, validity,
    and summaries. These methods aggregate information from the remote R sessions and return it in a structured format
    that can be used by the client application.
    """

    def __init__(self, dssession: DSSession):
        """
        Service for performing statistical operations on DataSHIELD sessions.

        Args:
            dssession: The DataSHIELD session to use for performing operations
        """
        self.dssession = dssession

    def get_classes(self, symbol: str) -> dict[str, list[str]]:
        """
        Get the classes of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the classes of in the remote R sessions
        Returns:
            A dictionary mapping server names to the classes of the specified symbol in the remote R sessions
        """
        classes = self.dssession.aggregate(f"classDS('{symbol}')")
        logger.debug(f"[{self.dssession.id}] Class for symbol '{symbol}': {classes}")
        # Make sure classes are lists (in case of single class, it might be returned as a string)
        for server, cls in classes.items():
            if isinstance(cls, str):
                classes[server] = [cls]
        return classes

    def get_length(self, symbol: str) -> dict[str, int]:
        """
        Get the length of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the length of in the remote R sessions
        Returns:
            A dictionary mapping server names to the length of the specified symbol in the remote R sessions
        """
        lengths = self.dssession.aggregate(f"lengthDS('{symbol}')")
        logger.info(f"[{self.dssession.id}] Length for symbol '{symbol}': {lengths}")
        return lengths

    def get_levels(self, symbol: str) -> dict[str, list[str]]:
        """
        Get the levels of a factor symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the levels of in the remote R sessions
        Returns:
            A dictionary mapping server names to the levels of the specified factor symbol in the remote R sessions
        """
        levels = self.dssession.aggregate(f"levelsDS({symbol})")
        logger.info(f"[{self.dssession.id}] Levels for symbol '{symbol}': {levels}")
        return levels

    def get_dimensions(self, symbol: str) -> dict[str, list[int]]:
        """
        Get the dimensions of a data.frame or matrix symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the dimensions of in the remote R sessions
        Returns:
            A dictionary mapping server names to the dimensions of the specified data.frame or matrix symbol in the remote R sessions
        """
        dims = self.dssession.aggregate(f"dimDS('{symbol}')")
        logger.info(f"[{self.dssession.id}] Dimensions for symbol '{symbol}': {dims}")
        # Make sure dimensions are lists of integers
        for server, dim in dims.items():
            if dim is None:
                dims[server] = []
        return dims

    def get_frequencies(self, symbol: str) -> dict[str, Any]:
        """
        Get the frequencies of a factor or logical symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the frequencies of in the remote R sessions
        Returns:
            A dictionary mapping server names to the frequencies of the specified factor or logical symbol in the remote R sessions
        """
        frequencies = self.dssession.aggregate(f"table1DDS({symbol})")
        logger.info(f"[{self.dssession.id}] Frequencies for symbol '{symbol}': {frequencies}")
        return frequencies

    def get_crosstab(self, symbol_x: str, symbol_y: str) -> dict[str, Any]:
        """
        Get the crosstab frequencies of two factor or logical symbols in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol_x: The first symbol name to get the crosstab frequencies of in the remote R sessions
            symbol_y: The second symbol name to get the crosstab frequencies of in the remote R sessions
        Returns:
            A dictionary mapping server names to the crosstab frequencies of the specified factor or logical symbols in the remote R sessions
        """
        crosstab = self.dssession.aggregate(f"table2DDS({symbol_x}, {symbol_y})")
        logger.info(f"[{self.dssession.id}] Crosstab frequencies for symbols '{symbol_x}' and '{symbol_y}': {crosstab}")
        return crosstab

    def get_quantile_means(self, symbol: str) -> dict[str, Any]:
        """
        Get the quantiles and mean of a numeric or integer symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the quantiles and mean of in the remote R sessions
        Returns:
            A dictionary mapping server names to the quantiles and mean of the specified numeric or integer symbol in the remote R sessions
        """
        quantile_means = self.dssession.aggregate(f"quantileMeanDS({symbol})")
        logger.info(f"[{self.dssession.id}] Quantiles and means for symbol '{symbol}': {quantile_means}")
        return quantile_means

    def get_names(self, symbol: str) -> dict[str, list[str]]:
        """
        Get the names of a list symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the names of in the remote R sessions
        Returns:
            A dictionary mapping server names to the names of the specified list symbol in the remote R sessions
        """
        names = self.dssession.aggregate(f"namesDS('{symbol}')")
        logger.info(f"[{self.dssession.id}] Names for symbol '{symbol}': {names}")
        return names

    def is_valid(self, symbol: str) -> dict[str, bool]:
        """
        Check the validity of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to check the validity of in the remote R sessions
        Returns:
            A dictionary mapping server names to the validity of the specified symbol in the remote R sessions
        """
        validity = self.dssession.aggregate(f"isValidDS({symbol})")
        logger.info(f"[{self.dssession.id}] Validity for symbol '{symbol}': {validity}")
        return validity

    def get_summary(self, symbol: str) -> dict[str, Any]:
        """
        Get the summary of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the summary of in the remote R sessions
        Returns:
            A dictionary mapping server names to the summary of the specified symbol in the remote R sessions
        """
        classes = self.get_unique_classes(symbol)

        validity = self.is_valid(symbol)

        if "data.frame" in classes or "matrix" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is a data.frame or matrix")
            dims = self.get_dimensions(symbol)
            cols = self.dssession.aggregate(f"colnamesDS('{symbol}')")
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "dimensions": {
                        "rows": dims[server][0] if len(dims[server]) > 0 else 0,
                        "columns": dims[server][1] if len(dims[server]) > 1 else 0,
                    },
                    "columns": cols[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        if "character" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is character")
            length = self.get_length(symbol)
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "length": length[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        if "factor" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is factor")
            length = self.get_length(symbol)
            levels = self.get_levels(symbol)
            frequencies = self.get_frequencies(symbol)
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "length": length[server],
                    "levels": levels[server],
                    "frequencies": frequencies[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        if "numeric" in classes or "integer" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is numeric or integer")
            length = self.get_length(symbol)
            quantile_means = self.get_quantile_means(symbol)
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "length": length[server],
                    "quantile_means": quantile_means[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        if "list" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is list")
            length = self.get_length(symbol)
            names = self.get_names(symbol)
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "length": length[server],
                    "names": names[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        if "logical" in classes:
            logger.debug(f"[{self.dssession.id}] Symbol '{symbol}' is logical")
            length = self.get_length(symbol)
            frequencies = self.get_frequencies(symbol)
            summaries = {}
            for server in validity:
                summaries[server] = {
                    "validity": validity[server],
                    "length": length[server],
                    "frequencies": frequencies[server],
                }
            logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
            return summaries

        summaries = {}
        for server in validity:
            summaries[server] = {
                "validity": validity[server],
            }
        logger.debug(f"[{self.dssession.id}] Summary for symbol '{symbol}': {summaries}")
        return summaries

    def get_mean(self, symbol: str) -> dict[str, Any]:
        """
        Get the mean of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the mean of in the remote R sessions
        Returns:
            A dictionary mapping server names to the mean value of the specified symbol in the remote R sessions
        """
        means = self.dssession.aggregate(f"meanDS({symbol})")
        logger.info(f"[{self.dssession.id}] Mean for symbol '{symbol}': {means}")
        return means

    def get_unique_classes(self, symbol: str) -> list[str]:
        """
        Get the unique classes of a symbol in the remote R sessions for a given DataSHIELD session.

        Args:
            symbol: The symbol name to get the unique classes of in the remote R sessions
        Returns:
            The unique classes of the specified symbol in the remote R sessions
        Raises:
            ValueError: If the specified symbol has multiple classes across servers
        """
        classes = self.get_classes(symbol)
        # Make sure class is unique accross servers
        the_classes = set()
        for _, cls in classes.items():
            for c in cls:
                the_classes.add(c)
        return list(the_classes)
