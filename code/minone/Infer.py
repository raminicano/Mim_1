from tqdm import tqdm
import torch

def inference(model, test_loader, device):
    # 모델을 지정한 디바이스로 이동합니다.
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for sentence, attention_mask in tqdm(test_loader):
            sentence = sentence.to(device)
            mask = attention_mask.to(device)

            # 모델을 통해 예측을 수행합니다.
            cls_output, pred = model(sentence, mask)

            # 예측 결과를 CPU로 이동하고 리스트로 변환하여 저장합니다.
            preds += pred.argmax(1).detach().cpu().numpy().tolist()

    return cls_output, preds
