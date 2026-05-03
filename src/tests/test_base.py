import uuid

from datashield import DSLoginBuilder, DSSession, DSError
import pytest
from datashield_base import StatsClient


class TestClass:
    @classmethod
    def setup_class(cls):
        # url = 'http://localhost:8080'
        url = "https://opal-demo.obiba.org"
        builder = DSLoginBuilder().add("server1", url, "dsuser", "P@ssw0rd", profile="survival")
        logins = builder.build()
        session = DSSession(logins)
        session.open()
        session.assign_table("df", tables={"server1": "CNSIM.CNSIM1"})
        print(session.has_errors())
        session_id = str(uuid.uuid4())
        cls.dssession = session

    @classmethod
    def teardown_class(cls):
        cls.dssession.close()

    @pytest.mark.integration
    def test_get_length(self):
        try:
            stats_service = StatsClient(self.dssession)
            lengths = stats_service.get_length("df")
            assert lengths == {"server1": 11}  # 11 columns in the df table
        except DSError as e:
            pytest.fail(f"get_length raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_col_length(self):
        try:
            stats_service = StatsClient(self.dssession)
            lengths = stats_service.get_length("df$LAB_HDL")
            assert lengths == {"server1": 2163}
        except DSError as e:
            pytest.fail(f"get_length raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_dimensions(self):
        try:
            stats_service = StatsClient(self.dssession)
            dimensions = stats_service.get_dimensions("df")
            assert dimensions == {"server1": [2163, 11]}
        except DSError as e:
            pytest.fail(f"get_dimensions raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_frequencies(self):
        try:
            stats_service = StatsClient(self.dssession)
            frequencies = stats_service.get_frequencies("df$GENDER")
            assert frequencies == {
                "server1": {"message": "valid Table", "table": [{"0": 1092, "1": 1071, "Total": 2163}]}
            }
        except DSError as e:
            pytest.fail(f"get_frequencies raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_quantile_means(self):
        try:
            stats_service = StatsClient(self.dssession)
            quantile_means = stats_service.get_quantile_means("df$LAB_HDL")
            assert quantile_means == {
                "server1": [
                    0.87524,
                    1.0474,
                    1.3,
                    1.581,
                    1.8445,
                    2.09,
                    2.2109,
                    1.56941631558514,
                ]
            }
        except DSError as e:
            pytest.fail(f"get_quantile_means raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_mean(self):
        try:
            stats_service = StatsClient(self.dssession)
            mean = stats_service.get_mean("df$LAB_HDL")
            assert mean == {
                "server1": {
                    "EstimatedMean": 1.56941631558514,
                    "Nmissing": 360,
                    "Ntotal": 2163,
                    "Nvalid": 1803,
                    "ValidityMessage": "VALID ANALYSIS",
                }
            }
        except DSError as e:
            pytest.fail(f"get_mean raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_is_valid(self):
        try:
            stats_service = StatsClient(self.dssession)
            validity = stats_service.is_valid("df")
            assert validity == {"server1": True}
        except DSError as e:
            pytest.fail(f"is_valid raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_summary(self):
        try:
            stats_service = StatsClient(self.dssession)
            summary = stats_service.get_summary("df")
            assert summary == {
                "server1": {
                    "columns": [
                        "LAB_TSC",
                        "LAB_TRIG",
                        "LAB_HDL",
                        "LAB_GLUC_ADJUSTED",
                        "PM_BMI_CONTINUOUS",
                        "DIS_CVA",
                        "MEDI_LPD",
                        "DIS_DIAB",
                        "DIS_AMI",
                        "GENDER",
                        "PM_BMI_CATEGORICAL",
                    ],
                    "dimensions": {
                        "rows": 2163,
                        "columns": 11,
                    },
                    "validity": True,
                },
            }
        except DSError as e:
            pytest.fail(f"get_summary raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_continuous_summary(self):
        try:
            stats_service = StatsClient(self.dssession)
            summary = stats_service.get_summary("df$LAB_HDL")
            assert summary == {
                "server1": {
                    "length": 2163,
                    "quantile_means": [
                        0.87524,
                        1.0474,
                        1.3,
                        1.581,
                        1.8445,
                        2.09,
                        2.2109,
                        1.56941631558514,
                    ],
                    "validity": True,
                },
            }
        except DSError as e:
            pytest.fail(f"get_continuous_summary raised an exception: {e} {self.dssession.get_errors()}")

    @pytest.mark.integration
    def test_get_categorical_summary(self):
        try:
            stats_service = StatsClient(self.dssession)
            summary = stats_service.get_summary("df$GENDER")
            assert summary == {
                "server1": {
                    "frequencies": {
                        "message": "valid Table",
                        "table": [
                            {
                                "0": 1092,
                                "1": 1071,
                                "Total": 2163,
                            },
                        ],
                    },
                    "length": 2163,
                    "levels": {
                        "Levels": ["0", "1"],
                        "ValidityMessage": "VALID ANALYSIS",
                    },
                    "validity": True,
                },
            }
        except DSError as e:
            pytest.fail(f"get_categorical_summary raised an exception: {e} {self.dssession.get_errors()}")
