import base64
import pytest
from unittest.mock import MagicMock, patch

from datashield import DSLoginBuilder, DSSession, DSError
from datashield_base import PlotsClient


# Minimal histogramDS2 response for a single server
_HIST_DS2_SERVER1 = {
    "value": [
        {
            "value": [
                {"value": [1.0, 2.0, 3.0, 4.0, 5.0]},  # breaks
                {"value": [10, 20, 15, 5]},  # counts
            ]
        }
    ]
}


class TestPlotsClientUnit:
    def setup_method(self):
        self.mock_session = MagicMock()
        self.mock_session.id = "test-session-id"
        self.client = PlotsClient(self.mock_session)

    def test_get_histogram_returns_valid_png_base64(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [0.0, 4.0]},
            {"server1": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            result = self.client.get_histogram("df$LAB_HDL")

        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_get_histogram_saves_file_to_work_dir(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [0.0, 4.0]},
            {"server1": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            self.client.get_histogram("df$LAB_HDL")

        expected = tmp_path / ".datashield" / "work" / "test-session-id" / "histogram_df_LAB_HDL.png"
        assert expected.exists()

    def test_get_histogram_calls_histogramds1_with_correct_args(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [0.0, 4.0]},
            {"server1": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            self.client.get_histogram("df$LAB_HDL", num_breaks=10, k=5, noise=0.1)

        first_call = self.mock_session.aggregate.call_args_list[0][0][0]
        assert "histogramDS1(df$LAB_HDL" in first_call
        assert "k=5" in first_call
        assert "noise=0.1" in first_call

    def test_get_histogram_calls_histogramds2_with_correct_args(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [0.0, 4.0]},
            {"server1": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            self.client.get_histogram("df$LAB_HDL", num_breaks=10, k=5, noise=0.1)

        second_call = self.mock_session.aggregate.call_args_list[1][0][0]
        assert "histogramDS2(df$LAB_HDL" in second_call
        assert "num.breaks=10" in second_call
        assert "k=5" in second_call
        assert "noise=0.1" in second_call

    def test_get_histogram_raises_when_range_is_none(self):
        self.mock_session.aggregate.return_value = {"server1": None}

        with pytest.raises(ValueError, match="Could not determine min and max"):
            self.client.get_histogram("df$LAB_HDL")

    def test_get_histogram_raises_when_range_is_empty(self):
        self.mock_session.aggregate.return_value = {"server1": []}

        with pytest.raises(ValueError, match="Could not determine min and max"):
            self.client.get_histogram("df$LAB_HDL")

    def test_get_histogram_uses_global_min_max_across_servers(self, tmp_path):
        # server2 has a wider range than server1
        self.mock_session.aggregate.side_effect = [
            {"server1": [2.0, 4.0], "server2": [1.0, 6.0]},
            {"server1": _HIST_DS2_SERVER1, "server2": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            self.client.get_histogram("df$LAB_HDL")

        second_call = self.mock_session.aggregate.call_args_list[1][0][0]
        # min=1.0 (integer) → stays 1; max=6.0 (integer) → becomes 7
        assert "min=1" in second_call
        assert "max=7" in second_call

    def test_get_histogram_multiple_servers_produces_valid_png(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [1.0, 4.0], "server2": [2.0, 5.0]},
            {"server1": _HIST_DS2_SERVER1, "server2": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            result = self.client.get_histogram("df$LAB_HDL")

        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_get_histogram_sanitizes_symbol_in_filename(self, tmp_path):
        self.mock_session.aggregate.side_effect = [
            {"server1": [0.0, 4.0]},
            {"server1": _HIST_DS2_SERVER1},
        ]
        with patch("datashield_base.plots.Path.cwd", return_value=tmp_path):
            self.client.get_histogram("df$col/sub")

        expected = tmp_path / ".datashield" / "work" / "test-session-id" / "histogram_df_col_sub.png"
        assert expected.exists()


class TestPlotsClientIntegration:
    @classmethod
    def setup_class(cls):
        url = "https://opal-demo.obiba.org"
        builder = DSLoginBuilder().add("server1", url, "dsuser", "P@ssw0rd", profile="survival")
        logins = builder.build()
        session = DSSession(logins)
        session.open()
        session.assign_table("df", tables={"server1": "CNSIM.CNSIM1"})
        cls.dssession = session

    @classmethod
    def teardown_class(cls):
        cls.dssession.close()

    @pytest.mark.integration
    def test_get_histogram_returns_png(self):
        try:
            client = PlotsClient(self.dssession)
            result = client.get_histogram("df$LAB_HDL")
            assert isinstance(result, str)
            decoded = base64.b64decode(result)
            assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
        except DSError as e:
            pytest.fail(f"get_histogram raised an exception: {e} {self.dssession.get_errors()}")
