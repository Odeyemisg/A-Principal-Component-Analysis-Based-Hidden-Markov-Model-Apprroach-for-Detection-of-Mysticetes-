clc
clear all
[X_DATA,fs_1]=audioread('940302-1222.wav'); %Loading the sound file
o_v=300;  %overlapping
syn=3000;%synchronising factor
state_num=4; %number of state
mix_num=2;  %number of mixture model
Coeff=12; %number of filter coefficient
 %%%%%%%%%%HMM_DATA_TRAIN%%%%%%%%%%%
  Temp=[45440 48840;67960 71560;106000 114080;140240 144280;181680 190400;236920 241200;243960 249480;339160 343200;345560 350440;408240 414000;449920 459920;462040 468200;499640 505840;509440 515800;557280 565120;605920 625480;629760 655680;659280 684360;1066760 1072080;1077400 1110320;1307160 1319080;1319920 1326920;1373040 1379440;1386440 1396440;1399400 1405560;1476560 1483160;1566040 1574760;1577760 1584120;1657880 1663840;1673200 1681480;1763320 1769680;1813680 1819640;1827280 1837080;1929320 1937800;1944640 1950560;2002000 2012440;2038800 2046000;2054520 2064080;2108720 2116360;2125080 2133600;2179480 2183760;2208840 2215200;2243680 2248360;2265360 2271960;2337200 2342760;2389080 2393960;2490480 2493640;2517040 2521280;2544440 2547840;2551880 2555920];%initialising sound positions

  for ii=1 :size(Temp,1)
     VV=Temp(ii,:); 
        S=VV(1);
        E=VV(2);
        U=X_DATA(S:E);
 D{ii}=U;
  end
for  ii =1: size(D,2)
      XX=cell2mat(D(ii));
      L=length(XX);
       KK=[];
      u=0;
     for jj=o_v:o_v:L
     if L-jj <o_v
            y=XX(u+1:end);
     else
         y=XX(1+u:jj);
     end 
     VV=lpc(y,Coeff); %%%extracting features with LPC
     VV=VV(:,2:end);
     KK=[KK; VV];
     u=u+o_v;
     end
     Data{ii}=KK;
 end
 Data;
 [p_start_data, A_data, phi_data, loglik_data] = ChmmGmm(Data, state_num, mix_num); %%%%%%%%sound HMM Training

%%%%%%%%HMM_NOISE_TRAIN%%%%%%%%%
 Temp=[10800 13000;20000 22000;56700 59000;72000 74500;79500 82000;95000 97500;103000 105500;127000 130000;153000 161500;176500 181000;225500 228000;265000 267000;326000 329000;400000 408000;482000 484800;493000	499000;542000 549000;591000	596800;774000 779500;805200	812000;880900 885000;904500	911300;918500 937000;999000	1017000;1111000	1119000;1125000	1141500;1152000	1155800;1458500 1464800;1483500 1489200;1584500 1592000;1611300 1618000;1628500 1631600;1691200 1698000;1795000 1801900;1872000 1878560;1897000 1904140;1938100 1944300;1993800 2001700;2080500 2085900;2116660 2124700;2184000 2190500;2199980 2208540;2215500 2221200;2248660 2253380;2272260 2295800;2377060 2381340;2394260 2398780;2433180 2459540;2464360 2471900;2556310	2561770]; %initialising sound positions


     for ii=1 :size(Temp,1)
     VV=Temp(ii,:); 
        S=VV(1);
        E=VV(2);
        U=X_DATA(S:E);
 D{ii}=U;
     end
  for  ii =1: size(D,2)
      XX=cell2mat(D(ii));
      L=length(XX);
       KK=[];
      u=0;
     for jj=o_v:o_v:L
     if L-jj <o_v
            y=XX(u+1:end);
     else
         y=XX(1+u:jj);
     end 
     VV=lpc(y,Coeff); %%%extracting features with LPC
     VV=VV(:,2:end);
     KK=[KK; VV];
     u=u+o_v;
     end
     Data{ii}=KK;
 end

  [p_start_noise, A_noise, phi_noise, loglik_noise] = ChmmGmm(Noise, state_num, mix_num); %%%%%%Noise HMM Training

%%%%%Decoding%%%%%%%%
%%%Combining the two HMM:Data and Noise%%%%%
p_start=[p_start_noise,p_start_data]; %%combining the start probaility

%%%Combined transition matrix %%%
A_C= zeros(state_num*2,state_num*2)  ; 
A_C(1:state_num,1:state_num)=A_noise;  %%Adding data transition
A_C(state_num+1:end,state_num+1:end)=A_data;  %%%Adding noise transition

%%%%% Transition switching%%%%
A_C(1,state_num+1)=0.5;
A_C(state_num+1,1)=0.5;

%%%%%%Adding transition from noise_to_data or data_to_noise %%%%%%%%%
%%%%%%%Combining mu,B and sigma for data and noise %%%%%%
DD=struct2cell(phi_data);
NN=struct2cell(phi_noise);
%%%%%%combine B
PP.B=[NN{1},DD{1}];
%%%%%%combine mu
PP.mu=cat(3,NN{2},DD{2});
%%combined sigma
PP.Sigma=cat(4,NN{3},DD{3});
%%%%%%Detecting%%%%%%%
Test_file=Extract_lpc(X_DATA,o_v,Coeff);%Extracting feature from unknown sound file
logp_xn_given_zn = Gmm_logp_xn_given_zn_test(Test_file,PP); %%%%%converting the Guassian variable
 path_data= LogViterbiDecode(logp_xn_given_zn,p_start,A_C); %%%%%%%%%Creating path for noise and sound
 Recog=Recognise(path_data,o_v,state_num,syn);

