import pytest
import torch

from colpali_engine.loss.colpali_2_losses import ColPali2Loss, ColPali2ModelOutput


@pytest.fixture
def single_vector_embeddings():
    return torch.randn(4, 128)


@pytest.fixture
def multi_vector_embeddings():
    return torch.randn(4, 10, 128)


@pytest.fixture
def model_output(single_vector_embeddings, multi_vector_embeddings):
    return ColPali2ModelOutput(single_vec_emb=single_vector_embeddings, multi_vec_emb=multi_vector_embeddings)


def test_single_vector_loss(single_vector_embeddings):
    loss_fn = ColPali2Loss()
    query_embeddings = single_vector_embeddings
    doc_embeddings = single_vector_embeddings

    outputs = loss_fn.single_vector_loss(query_embeddings, doc_embeddings)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])
    assert outputs.scores is None


def test_multi_vector_loss(multi_vector_embeddings):
    loss_fn = ColPali2Loss()
    query_embeddings = multi_vector_embeddings
    doc_embeddings = multi_vector_embeddings

    outputs = loss_fn.multi_vector_loss(query_embeddings, doc_embeddings)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])
    assert outputs.scores is None


def test_distillation_loss():
    loss_fn = ColPali2Loss()
    teacher_scores = torch.randn(4, 4)
    student_scores = torch.randn(4, 4)

    outputs = loss_fn.distillation_loss(teacher_scores, student_scores)

    assert outputs.loss is not None
    assert outputs.loss.shape == torch.Size([])


def test_forward(model_output):
    loss_fn = ColPali2Loss()
    query_embeddings = model_output
    doc_embeddings = model_output

    outputs = loss_fn.forward(query_embeddings, doc_embeddings)

    assert outputs.single_vector_loss is not None
    assert outputs.multi_vector_loss is not None
    assert outputs.total_loss is not None
    if loss_fn.use_distillation_loss:
        assert outputs.distillation_loss is not None
