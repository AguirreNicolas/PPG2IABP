function [Ind_inflex]=FUN_ANAL_AIx_modificado(Sen_P,left,right,SBP,sys_idx,DBP,Modo_Det,VerCiclos)
%{
Entrada: 
       SEN_P     -> Ciclo Presion
       left      -> Indice de inicio del ciclo
       right     -> Indice de fin de ciclo
       Modo_Det  -> Modo de detecci�n del segundo pico s
                0 Por Se�al
                1 Por Derivada Segunda
                2 Por Derivada Cuarta 
       VerCiclos -> Verificaci�n de puntos caracter�sticos ciclo a ciclo
Salida:
       Ti          -> Indice de Pi

En la derivada de CUARTO ORDEN de la onda de Presi�n Arterial se generan 
onditas (tipo senos) donde:
  El pie de onda (Presi�n Diast�lica) est�  en el primer cruce por cero yendo de 
  + a - (mitad de la primera ondita)
  El hombro (Onda reflejada) est� en el segundo cruce por cero yendo de 
  + a - (mitad de la segunda ondita)
  El pico sit�lico (Presi�n Sist�lica) est� en el segundo cruce por cero yendo
de + a - (principio de la tercera ondita)
  El punto de inflexi�n esta en el segundo pico de la derivada de SEGUNDO
orden
NOTA: Si se usa s�lo derivada de 4to orden puede haber errores en la
detecci�n del punto de inflexi�n en ondas tipo D, pues en este tipo de 
ondas el hombro y el punto de inflexi�n no coinciden (adultos mayores alta
rigidez). Por eso se complementa con la de 2do orden (SISTEMA SPHYGMOCOR)


--------------------------------------------------------------------------
%}
%{
Sen_P = smooth(Sen_P_o);
figure();
subplot(2,1,1)
plot(Sen_P_o)
subplot(2,1,2)
plot(Sen_P)
%}
%Modo visualizaci�n de ciclos o barra de espera
if VerCiclos==1
  h=figure;
  %set(gcf,'units','normalized','outerposition',[0 0 1 1])
end

%El an�lisis se efect�a a partir de la derivad� m�xima del ciclo (creciemiento)
%para poder detectar correctamente el punto de inflexci�n por derivada segunda
%Esto evita problemas si la detacci�n de los ciclos se gener� por
%distintos m�todos.
[~, Ind_IniAnal]=max(gradient(Sen_P));
if Ind_IniAnal > sys_idx
    Ind_inflex = [];
    return
end
SEN_Pc = Sen_P(Ind_IniAnal+1:end); 
%tc = (left + Ind_IniAnal):right;
tc = 1:right - Ind_IniAnal + 1;
%C�lculo de los puntos m�ximo, m�nimo e inflexi�n
%Valor Presi�n SIST�LICA
V_max = SBP;
Ind_max = sys_idx - Ind_IniAnal + 1;

%Valor Presi�n DIAST�LICA
V_min = DBP;
%Ind_min = right-(left+Ind_IniAnal);


%% Main
%Valor Segundo SEGUNDO PICO DEL CICLO (INFLEXI�N)
switch Modo_Det
  case 0 %Detecci�n del segundo pico del ciclo directamente sobre la se�al
    [Picos Ind_Picos]=findpeaks(filtfilt(ones(1,20)/20,1,SEN_Pc));
    if length(Ind_Picos)>2
      Ind_inflex=Ind_Picos(2);
      V_inflex=SEN_Pc(Ind_Picos(2));
    elseif length(Ind_Picos)==1 
      Ind_inflex=Ind_Picos(1);
      V_inflex=SEN_Pc(Ind_Picos(1));
    else
      Ind_inflex=1;
      V_inflex=SEN_Pc(1);
    end     
  case 1 %Detecci�n del segundo pico del ciclo por derivada segunda
     %Derivada segundo orden y picos
     SEN_P_d2=gradient(gradient(SEN_Pc));
     [Val,Ind_Max_d2] = findpeaks(abs(SEN_P_d2));  
     Ind_inflex = Ind_Max_d2(2);
     %V_inflex = SEN_Pc(Ind_inflex);
      V_inflex = SEN_Pc(Ind_Max_d2(3)); %ORIGINAL, POR QUE INDICE 3?
     
  case 2 %Detecci�n del segundo pico del ciclo por derivada cuarta
     %Derivada de 4to orden y cruces por cero
     SEN_P_d4=gradient(gradient(gradient(gradient(SEN_Pc))));
     %Detecci�n de los cruces por cero
     Ind_Cruces=zeros(1,length(SEN_P_d4));
     for m=1:length(SEN_P_d4)-1
       %Condici�n del cruce por cero
       if (SEN_P_d4(m)<0 && SEN_P_d4(m+1)>0)||(SEN_P_d4(m)>0 && SEN_P_d4(m+1)<0) 
         Ind_Cruces(m)=m;
       end
     end
     Ind_Cruces=find(Ind_Cruces);
     %Segundo pico en 
     Ind_inflex=Ind_Cruces(3);
     V_inflex=Sen_P(Ind_inflex);
end
%Visualizaci�n de ciclos con posibilidad de correcci�n      
if VerCiclos==1
switch Modo_Det
    case 0 
    case 1 %Derivada 2do orden
      subplot(211),plot(tc,SEN_P_d2),grid on;axis tight,hold on;
      plot(tc(Ind_Max_d2),SEN_P_d2(Ind_Max_d2),'r.','Markersize',20),hold off;
      ylabel('d2P/dt');title('Derivada Segunda Se�al de Presi�n');hold off;
      title('Verifique Presi�n Inflexi�n: Segundo pico','fontweight','b');
      subplot(212);
    case 2 %Derivada 4to orden
      %Visualizaci�n derivada de cuarto orden
      subplot(211),plot(tc,SEN_P_d4),grid on;axis tight,hold on;
      ylabel('d4P/dt');title('Derivada Cuarta Se�al de Presi�n');
      plot(tc(Ind_Cruces),SEN_P_d4(Ind_Cruces),'r.','Markersize',20),hold off;
      title('Verifique Presi�n Inflexi�n: 2do cruce por cero + a - ','fontweight','b');
      subplot(212);
end    
%Puntos Caracter�sticos MAX, MIN, INFLEXI�N
plot(tc,SEN_Pc);grid on;axis tight;hold on;
%Valor M�ximo
plot(tc(Ind_max),V_max,'r.','Markersize',20);
%Valor M�nimo
plot(tc(end),V_min,'g.','Markersize',20); % Si es que calculo right o left
%VERIFICACI�N Velor Punto de Inflexi�n
plot(tc(Ind_inflex),SEN_Pc(Ind_inflex),'c.','Markersize',20),hold off;
title(['Presi�n de Inflexi�n Detectada'],'fontweight','b');
%legend('Presi�n Arterial',['Presi�n Sist�lica: ' num2str(V_max)],['Presi�n Diast�lica: ' num2str(V_min)],'Punto de Inflexi�n',1);
xlabel ('t [idx]');ylabel('Presi�n [mmHg]');hold off; 

%% Cuadro de Dialogo - Manual Selection
%{
ModValor=questdlg('�Modificar Punto de Inflexi�n?', ...
                  'Verificaci�n Punto de Inflexi�n', ...
                  'Aceptar','Modificar','Cancelar','Cancelar');
switch ModValor
    case 'Aceptar'
    case 'Modificar'
        [x,y]=ginput(1);
        %B�squeda del valor m�s cercano existente en el vector de abscisas
        %Distancia m�nima de x selecc. a un punto existente en el vector
        [Dist Ind_inflex]=min(abs(tc-x));
    case 'Cancelar'
        close(h);
        return

end
%}
end


%% Index Shift
%Estimaci�n del AIx=(Pmax-Pinflex)/(Pmax-Pmin)
Ti_g = left+Ind_IniAnal + Ind_inflex; % Con respecto a toda la señal
Pi = SEN_Pc(Ind_inflex);
Pi_2 = Sen_P(Ti_g - left);
Ind_inflex = Ind_inflex + Ind_IniAnal +1;

end