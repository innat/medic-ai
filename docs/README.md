# Docs Layout

This directory contains the source for the project documentation.

## Structure

- `docs/examples/`: end-to-end example pages grouped by task.
- `docs/guides/`: conceptual guides and API-oriented walkthroughs.
- `docs/getting_started/`: onboarding and setup pages.
- `docs/misc/`: contribution notes and FAQ pages.
- `docs/_static/`: theme-level static files such as CSS, logos, and favicon assets.
- `docs/assets/`: page content assets used inside documentation pages.

## Example Assets

Assets for example pages live under `docs/assets/examples/`.

Use the example file name as the asset folder name. For example:

- `docs/examples/segmentation/brain_tumor.md`
- `docs/assets/examples/brain_tumor/`

- `docs/examples/classification/medmnist_multiclass.md`
- `docs/assets/examples/medmnist_multiclass/`

This keeps example-specific images close to the docs tree without mixing them into the theme assets in `docs/_static/`.

## Notes

- Prefer `docs/assets/` for screenshots, figures, and other page content images.
- Prefer `docs/_static/` only for shared site assets such as logos, CSS, or favicon files.
- When adding a new example, create its matching asset directory before referencing images from the page.
