class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.update_step = 5 ## task-level inner update steps
        self.update_step_test = 5  
        self.net = BaseNet()
        self.meta_lr = 2e-4
        self.base_lr = 4 * 1e-2
        self.inner_lr = 0.4
        self.outer_lr = 1e-2
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)
        
    def forward(self,x_spt, y_spt, x_qry, y_qry):
        # 初始化
        task_num, ways, shots, h, w = x_spt.size()
        query_size = x_qry.size(1) # 75 = 15 * 5
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]
        
        for i in range(task_num):
            ## 第0步更新
            y_hat = self.net(x_spt[i], params = None, bn_training=True) # (ways * shots, ways)
            loss = F.cross_entropy(y_hat, y_spt[i]) 
            grad = torch.autograd.grad(loss, self.net.parameters())
            tuples = zip(grad, self.net.parameters()) ## 将梯度和参数\theta一一对应起来
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
            with torch.no_grad():
                y_hat = self.net(x_qry[i], self.net.parameters(), bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[0] += correct
            
            # 使用更新后的数据在query集上测试。
            with torch.no_grad():
                y_hat = self.net(x_qry[i], fast_weights, bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[1] += correct   
            
            for k in range(1, self.update_step):
                
                y_hat = self.net(x_spt[i], params = fast_weights, bn_training=True)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights) 
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                    
                y_hat = self.net(x_qry[i], params = fast_weights, bn_training = True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[k+1] += loss_qry
                
                with torch.no_grad():
                    pred_qry = F.softmax(y_hat,dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    correct_list[k+1] += correct
#         print('hello')
                
        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad() # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()
        
        accs = np.array(correct_list) / (query_size * task_num)
        loss = np.array(loss_list_qry) / ( task_num)
        return accs,loss

    
    
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4
        
        query_size = x_qry.size(0)
        correct_list = [0 for _ in range(self.update_step_test + 1)]
        
        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))
        
        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with torch.no_grad():
            y_hat = new_net(x_qry,  params = new_net.parameters(), bn_training = True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[0] += correct

        # 使用更新后的数据在query集上测试。
        with torch.no_grad():
            y_hat = new_net(x_qry, params = fast_weights, bn_training = True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params = fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
            
            y_hat = new_net(x_qry, fast_weights, bn_training=True)
            
            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry).sum().item()
                correct_list[k+1] += correct
                
        del new_net
        accs = np.array(correct_list) / query_size
        return accs  
