import pytest
import torch

from colpali_engine.loss.colpali_duo_losses import ColPaliDuoLoss, ColPaliDuoModelOutput

BATCH_SIZE = 4
EMBEDDING_DIM = 128
SEQ_LENGTH = 10


@pytest.fixture
def single_vector_embeddings() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, EMBEDDING_DIM)


@pytest.fixture
def multi_vector_embeddings() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM)


@pytest.fixture
def model_output(
    single_vector_embeddings: torch.Tensor,
    multi_vector_embeddings: torch.Tensor,
) -> ColPaliDuoModelOutput:
    return ColPaliDuoModelOutput(
        single_vec_emb=single_vector_embeddings,
        multi_vec_emb=multi_vector_embeddings,
    )


@pytest.fixture(
    params=[
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
)
def colpali_duo_loss(request) -> ColPaliDuoLoss:
    use_matryoshka_loss, use_distillation_loss = request.param
    return ColPaliDuoLoss(
        use_matryoshka_loss=use_matryoshka_loss,
        use_distillation_loss=use_distillation_loss,
        matryoshka_dims=[EMBEDDING_DIM, EMBEDDING_DIM // 2],
    )


def test_single_vector_loss(
    colpali_duo_loss: ColPaliDuoLoss,
    single_vector_embeddings: torch.Tensor,
):
    query_embeddings = single_vector_embeddings
    doc_embeddings = single_vector_embeddings

    outputs = colpali_duo_loss.single_vector_loss(query_embeddings, doc_embeddings)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])
    assert outputs.scores is None


def test_multi_vector_loss(
    colpali_duo_loss: ColPaliDuoLoss,
    multi_vector_embeddings: torch.Tensor,
):
    query_embeddings = multi_vector_embeddings
    doc_embeddings = multi_vector_embeddings

    outputs = colpali_duo_loss.multi_vector_loss(query_embeddings, doc_embeddings)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])
    assert outputs.scores is None


def test_distillation_loss(colpali_duo_loss: ColPaliDuoLoss):
    teacher_scores = torch.randn(BATCH_SIZE, BATCH_SIZE)
    student_scores = torch.randn(BATCH_SIZE, BATCH_SIZE)

    outputs = colpali_duo_loss.distillation_loss(teacher_scores, student_scores)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])


def test_forward(
    colpali_duo_loss: ColPaliDuoLoss,
    model_output: ColPaliDuoModelOutput,
):
    query_embeddings = model_output
    doc_embeddings = model_output

    outputs = colpali_duo_loss.forward(query_embeddings, doc_embeddings)

    assert outputs.single_vector_loss is not None
    assert outputs.multi_vector_loss is not None
    assert outputs.loss is not None
    if colpali_duo_loss.use_distillation_loss:
        assert outputs.distillation_loss is not None
