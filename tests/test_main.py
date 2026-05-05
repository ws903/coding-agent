from unittest.mock import AsyncMock, MagicMock, patch


@patch("agent.__main__.run_interactive", new_callable=AsyncMock)
@patch("agent.__main__.parse_args")
def test_main_interactive(mock_parse_args, mock_run_interactive):
    from agent.__main__ import main

    mock_parse_args.return_value = MagicMock(auto=False)
    main()
    mock_run_interactive.assert_called_once()


@patch("agent.__main__.sys")
@patch("agent.__main__.run_autonomous", new_callable=AsyncMock, return_value=0)
@patch("agent.__main__.parse_args")
def test_main_autonomous(mock_parse_args, mock_run_autonomous, mock_sys):
    from agent.__main__ import main

    mock_parse_args.return_value = MagicMock(auto=True)
    main()
    mock_run_autonomous.assert_called_once()
    mock_sys.exit.assert_called_once_with(0)


@patch("agent.__main__.sys")
@patch("agent.__main__.run_autonomous", new_callable=AsyncMock, return_value=1)
@patch("agent.__main__.parse_args")
def test_main_autonomous_failure(mock_parse_args, mock_run_autonomous, mock_sys):
    from agent.__main__ import main

    mock_parse_args.return_value = MagicMock(auto=True)
    main()
    mock_sys.exit.assert_called_once_with(1)
