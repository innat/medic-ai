# Contribution

Please refer to the current [roadmap](https://github.com/innat/medic-ai/wiki/Roadmap) for an overview of the project. Feel free to explore anything that interests you. If you have suggestions or ideas, please open a [GitHub issue](https://github.com/innat/medic-ai/issues/new/choose) so it can be discussed.

1. Install `medicai` from source:

```bash
git clone https://github.com/innat/medic-ai
cd medic-ai
pip install keras -qU
pip install -e .
```

Add your contribution and implement relevant test code.

2. Run tests:

```bash
python -m pytest test/

# or run only a specific test selection
python -m pytest -k new_method
```
