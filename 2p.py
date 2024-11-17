#기본설정
import numpy as np #연산과 수학 함수에 사용하는 임포트

#랜덤한 기울기, y절편 설정과 스텝사이즈 결정
gradient, yint = np.random.rand(2) #gradient,yint를 0~1사이의 수로 초기화(gradient:기울기,bais:y절편)
lr = 0.01 #학습률(스텝 사이즈)

#학습 데이터 제공(y=2x+2)
x = np.array([1,3,5,7,9,11]) #입력값(x)
y = np.array([4, 8, 12, 16, 20, 24]) #출력값(y)

#반복과 학습
for i in range(1000): #1000번 반복
    pred = gradient * x + yint #입력값(x)에 대한 예측값 설정
    error = ((y - pred)**2).mean() #y_train(실제 값)과 pred(예측 값)의 차이를 제곱하여 평균 구함(오차)
    gradient = gradient - lr * ((pred - y) * x).mean() #스텝 사이즈를 기울기에 맞게 조정
    yint = yint - lr * (pred - y).mean() #위 방식과 비슷한 방식으로 y절편 조정
    
    #결과물 출력
    print(f'rep : {i+1}, gradient : {gradient:.4f}, yint : {yint:.4f}, error : {error:.4f}') #매 반복마다 결과값 출력