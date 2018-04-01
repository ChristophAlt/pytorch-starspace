import click

from src.utils import get_fields


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True)
@click.option('--dataset_format', type=click.Choice(['ag_news']), default=None)
@click.option('--test-file', type=click.Path(exists=True), required=True)
def test(model, dataset_format, test_file):
    lhs_field, rhs_field = get_fields(dataset_format)
    
    test_dataset, extractor_func = get_dataset_extractor(test_file, dataset_format, lhs_field, rhs_field)
    