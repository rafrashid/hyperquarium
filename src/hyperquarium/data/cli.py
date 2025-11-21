import click


@click.group('data')
def cli():
    """
    Data processing operations
    """
    pass


@cli.command('load_cube')
@click.argument('definition', type=click.Path())
@click.option('--data-root', type=click.Path(file_okay=False, exists=True))
@click.option('--data-file', type=click.Path(dir_okay=False, exists=True))
def cli_load_cube(definition, data_root, data_file):
    """
    Load data cube
    """
    from .cube_loader import CubeLoader
    loader = CubeLoader(definition, data_root)

    if data_file:
        click.echo(f'Loading {data_file}')
        cube = loader.load(data_file)
        print(cube)
