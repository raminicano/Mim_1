from tqdm import tqdm
import torch


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for sentence, attention_mask in tqdm(test_loader): 
            sentence = sentence.to(device)
            mask = attention_mask.to(device)

            cls_output, pred = model(sentence, mask)

            pred = pred.argmax(1).detach().cpu().numpy().tolist()

    return cls_output, pred