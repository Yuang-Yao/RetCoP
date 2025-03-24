"""
 Downstream task adapter initialization, training, prediction
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import math
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from dataloader.preprocessing import augmentations_pretraining


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
 
pure vision adapter uses only the image encoder
"""
# Initialization of the adapter's parent class;
# Implement training interface; Training and predicting virtual functions; Extract visual features and labels
class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        '''
        targets：  category     tta：  test time data augmentation   fta：  train/fit time data augmentation
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()                               #   Freezing encoder parameter
        self.num_targets = len(targets)                 #   Number of classes
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20                  #   enhancement times for fta/tta

    #   Get visual features and labels
    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        #  CLIP 
        # adapter input is the feature vector of CLIP visual encoder and the output is the class number
        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)

                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # Training interface:  data augmentation, extract visual features, call training functions
    def fit(self, loaders, transforms=None, dataset=None):
        data_loader = loaders["train"]                                                         #   img path /mask

        #    use augmentation strategies to increase training data?
        if self.fta:
            transforms = augmentations_pretraining
        #   get visual features
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)                                                       #   Merge into one dimension
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        self.train(X, Y, dataset)

    #   Virtual functions for training
    def train(self, X, Y, dataset):
        """
           Implemented by a specific adapter
        """
        return

    #   Virtual functions for prediction
    def predict(self, loader, transforms=None):
        """
           Implemented by a specific adapter
        """
        return


# LP        Logistic regression training; forecast   LP
class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        '''
        c  Regularization coefficient of logistic regression
        '''
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0, class_weight="balanced")

    # LP    The parameters of LP are trained and added to the model
    def train(self, X, Y, dataset=None):
        '''
        X  image features  Y（） label
        '''
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)

        if not dataset == 'Angiographic':
            self.classifier.fit(X, Y)

            # FLAIR   Add the trained logistic regression to the FLAIR model
            self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
            self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        else:
            # sklearn
            X = torch.tensor(X)
            Y = torch.tensor(Y, dtype=torch.float)      # 

            epochs = 10
            self.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True).to(device)

            optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=0.001, eps=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

            indexes = np.arange(0, X.shape[0])
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(X.shape[0]):
                    X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)  # 1 * embd
                    target = Y[indexes[i_sample]].unsqueeze(0).to(device)  # 1，

                    logits = self.classifier(X_batch)
                    loss_fuc = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fuc(logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_epoch += loss.item() / X.shape[0]
                print('loss=%2.5f' % loss_epoch, end="\n")

            self.model.classifier = self.classifier

        self.model.classifier.to(device)

    #   prediction
    def predict(self, loader, transforms=None, dataset=None):
        self.model.eval()
        # Data augmentation at test
        # Multiple augmentation to a sample and feed into the model to vote (mean)
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        with torch.no_grad():
            refs, preds = [], []                                            # refs gt     preds  prediction
            for step, batch in enumerate(epoch_iterator):
                #   data
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                #  logits  forward pass outputs logits
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))               # bs * K * 1
                    score = torch.concat(preds_tta, -1).mean(-1)            # bs * K * number_augmentations==》bs * K
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)

                #   activation function
                if score.shape[-1] == 1:
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)     # 2      bs * 2
                elif dataset == 'Angiographic':
                    score = torch.sigmoid(score)
                else:
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
   Multimodal adapter
"""
# Multimodal adapter parent class inherits the adapter parent class; Increase the extraction of text features
class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Input category name. Output text characteristics corresponding to the category (with/without domain knowledge)
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge)


# ZS
class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    #   Clear the operation of the training interface
    def fit(self, loaders, transforms=None, dataset=None):
        return

    def predict(self, loader, transforms=None, dataset = None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)                #   Get image features and labels
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()    #  compute similarity     X:N * embd      text_embeds：K * embd
                scores.append(score_i.unsqueeze(-1))                                                                # score_i.unsqueeze(-1)：N * K * 1
            score = torch.concat(scores, -1).mean(-1)                                                               # score=logits     N * K
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        if dataset == 'Angiographic':
            preds = torch.sigmoid(score)
        else:
            preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()
        return refs, preds


# CLIP   ；；   CLIP adapter: residuals adapter from visual space to text space; Training; forecast
class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim     #   Feature output dimension
        self.reduction = 4
        self.ratio = 0.2                                # adapter  residuals structure    ratio * adapter(x) + (1-ratio)  * x

        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None, dataset=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    X = self.residual_adapter(X)
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                X = self.residual_adapter(X)
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, dataset=None):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        epochs, lr, bs = 10, 0.001, 1
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])
        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)                           # 1 * embd
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)                               # 1，

                X_batch = self.residual_adapter(X_batch)
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)   #   compute similarity
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    #    Residuals adapter for visual space to text space
    def residual_adapter(self, X):
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)                                    #    Normalize features
        return X


# CLIP   ；；   CLIP adapter: residuals adapter from visual space to text space; Training; forecast
class ClipAdapter_Angio(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim     #   Feature output dimension
        self.reduction = 4
        self.ratio = 0.2                                # adapter  residuals structure    ratio * adapter(x) + (1-ratio)  * x

        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None, dataset=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    X = self.residual_adapter(X)
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                X = self.residual_adapter(X)
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, dataset=None):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        epochs, lr, bs = 10, 0.001, 1
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])
        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)                           # 1 * embd
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)                               # 1，
                target = target / target.sum(dim=-1, keepdim=True)  # 1
                target = target.float()  # 

                X_batch = self.residual_adapter(X_batch)
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)   #   compute similarity
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    #    Residuals adapter for visual space to text space
    def residual_adapter(self, X):
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)                                    #    Normalize features
        return X

class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        '''
        train：
        '''
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.cache_keys = []            #   visual feature vectors
        self.cache_values = []          #  one-hot  label one-hot vector

        #   train?
        self.train_tip = train
        self.adapter_layer = []         #   adapter

    def predict(self, loader, transforms=None, dataset=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)               # X: N * embed   test set data
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, dataset=None):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)              # embed * N'  （）  count similarity with itself (training set)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)                 # N' * K

        if self.train_tip:
            epochs, lr, bs = 1, 0.001, 1
            #       X：N' * embed      linear layer：embed ==》 N'
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())                              #  N' * embed  X，
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)
            indexes = np.arange(0, self.cache_keys.shape[1])                                            # self.cache_keys.shape[1]   number of sample
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)               # self.cache_keys：embed * N'  ==》 1 * embed
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)   # self.cache_values：N' * K          label index

                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())
                    affinity = adapter_layer(image)                                                     # 1 * embed ==》 1 * N'  （ ）  Initialize the weights with reference to the training set
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values   # 1 * K （）  Use the training set as a reference
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()
                    tip_logits = clip_logits + cache_logits * self.alpha
                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    #  +   Adapter + similarity calculation
    def adapter(self, X):
        # X：N * embed     Visual feature of the test set
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))           # N * K    similarity calculation

        if not self.train_tip:
            affinity = X @ self.cache_keys                                  # N * embed * embed * N' ==》 N * N'        Combine training data and test data
        else:
            affinity = self.adapter_layer(X)                                # embed ==》 N'   N * N'  output shape

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values       # N * N' * N' * K ==》 N * K    training sets' label as reference
        logits = clip_logits + cache_logits * self.alpha

        return logits

