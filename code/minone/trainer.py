import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_dataloader, val_dataloader, epochs, device, patience=2, gradient_accumulation_steps=3):
        # 모델과 옵티마이저 등 초기화
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Early Stopping을 위한 변수
        self.early_stopping_counter = 0
        self.best_loss = float('inf')
        self.best_model = None

    def train(self):
        # 모델을 GPU로 이동
        self.model.to(self.device)
        
        # 훈련 중의 손실, 정확도, F1 점수를 저장할 리스트 초기화
        train_loss_list, val_loss_list = [], []
        acc_list, f1_list = [], []

        for epoch in range(self.epochs):
            self.model.train()
            
            train_loss = []
            total_loss = 0

            for idx, (sentence, attention_mask, label) in enumerate(tqdm(iter(self.train_dataloader))):
                sentence = sentence.to(self.device)
                label = label.type(torch.LongTensor).to(self.device)
                mask = attention_mask.to(self.device)

                self.optimizer.zero_grad()

                # 모델에 입력 데이터 전달하여 예측 수행
                _, pred = self.model(sentence, mask)
                loss = self.criterion(pred, label)

                loss.backward()

                # 그래디언트 누적 후 업데이트
                if (idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    total_loss += loss.item()

                train_loss.append(loss.item())

            if idx % self.gradient_accumulation_steps != 0:
                self.optimizer.step()
                total_loss += loss.item()

            val_loss, val_f1, val_accuracy = self.validation()

            # 훈련 중의 손실 및 검증 중의 손실, 정확도, F1 점수 저장
            train_loss_list.append(np.mean(train_loss))
            val_loss_list.append(val_loss)
            acc_list.append(val_accuracy)
            f1_list.append(val_f1)

            # 에폭마다 결과 출력
            print(f'Epoch: [{epoch}] Train Loss: [{np.mean(train_loss):.5f}] Val Loss: [{val_loss:.5f}] F1: [{val_f1:.5f}] Accuracy: [{val_accuracy:.5f}]')

            # 검증 손실이 더 낮으면 모델 저장
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model = self.model
                torch.save(self.model.state_dict(), "./code/minone/model.pth")  
                print("Model parameters saved!")
                self.early_stopping_counter = 0  # 초기화
            else:
                self.early_stopping_counter += 1

            # patience 횟수만큼 검증 손실이 개선되지 않으면 조기 종료
            if self.early_stopping_counter >= self.patience:
                print(f'Early stopping triggered after {epoch} epochs without improvement.')
                break

        # 스케줄러 업데이트
        self.scheduler.step(val_loss)

        return train_loss_list, val_loss_list, acc_list, f1_list

    def validation(self):
        # 모델을 평가 모드로 설정
        self.model.eval()
        val_loss = []

        val_preds = []
        val_labels = []

        with torch.no_grad():
            for sentence, attention_mask, label in tqdm(iter(self.val_dataloader)):
                sentence = sentence.to(self.device)
                label = label.type(torch.LongTensor).to(self.device)
                mask = attention_mask.to(self.device)

                # 모델에 입력 데이터 전달하여 예측 수행
                _, pred = self.model(sentence, mask)
                loss = self.criterion(pred, label)

                val_loss.append(loss.item())

                val_preds += pred.argmax(1).detach().cpu().numpy().tolist()
                val_labels += label.detach().cpu().numpy().tolist()

        # F1 점수와 정확도 계산
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_accuracy = accuracy_score(val_labels, val_preds)

        return np.mean(val_loss), val_f1, val_accuracy
