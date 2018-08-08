function[miu_final,pi_final,vars_final]= Training(dimension,c,sample,pi_prev,vars_prev,miu_prev)
   trainingsample=sample(:,1:dimension);
   h_prev=zeros(size(trainingsample,1),c);
   h_curr=h_prev;
   miu_curr=miu_prev;
   pi_curr=pi_prev;
   vars_curr=vars_prev;
   times=0;
   vars_min=0.01*eye(dimension);
   while times<1
       for i=1:size(trainingsample,1)
            sum_1=0;
            for index=1:c
                sum_1=sum_1+mvnpdf(trainingsample(i,:),miu_prev(:,index)',vars_prev(:,:,index))*pi_prev(index);
            end
            for j=1:c
                h_curr(i,j)=mvnpdf(trainingsample(i,:),miu_prev(:,j)',vars_prev(:,:,j))*pi_prev(j)/sum_1;
            end
       end
       
       for j=1:c
           sum_1=0;
           sum_2=0;
           for i=1: size(trainingsample,1)
               sum_1=sum_1+h_curr(i,j).*trainingsample(i,:)';%sample=TrainsampleDCT_BG(:,1:dimension);
               sum_2=sum_2+h_curr(i,j);
           end
           miu_curr(:,j)=sum_1./sum_2;%miu_prev=rand(dimension,8);
           pi_curr(j)=sum_2/size(trainingsample,1);
       end
       
       for j=1:c
           sum_1=0;
           sum_2=0;
           for i=1: size(trainingsample,1)
               sum_1=sum_1+h_curr(i,j)*(trainingsample(i,:)-miu_curr(:,j)')'*(trainingsample(i,:)-miu_curr(:,j)');
               sum_2=sum_2+h_curr(i,j);
           end
           vars_curr(:,:,j)=sum_1/sum_2;
           vars_curr(:,:,j)=vars_curr(:,:,j).*eye(dimension);
           vars_curr(:,:,j)=max(vars_curr(:,:,j),vars_min);
       end
       
       %times=times+1;
%        if (sum(abs(pi_curr-pi_prev))<0.001) && 
%            break;
%        else
           h_prev=h_curr;
           miu_prev=miu_curr;
           vars_prev=vars_curr;
           pi_prev=pi_curr;
       end
   end
   
   miu_final=miu_curr;
   pi_final=pi_curr;
   vars_final=vars_curr;
end