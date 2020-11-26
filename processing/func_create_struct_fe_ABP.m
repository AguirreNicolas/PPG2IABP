function struct_feature_ABP = func_create_struct_fe_ABP(q)

    %Struct. of Features
    for i = q:-1:1
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %%%% DERIVADA MAXIMA Y MINIMA Y SUS INDICES
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Idx
        struct_feature_ABP(i).left = NaN; % Index Start of pulse
        struct_feature_ABP(i).right = NaN; % Index End of pulse
        struct_feature_ABP(i).sys_idx = NaN; %Index Systolic peak
        struct_feature_ABP(i).dicN_idx = NaN; % Index DNotch

        struct_feature_ABP(i).skew = NaN; %Skew
        %% Time Features
        struct_feature_ABP(i).iTTSP = NaN; % Time to Systolic Peak
        struct_feature_ABP(i).TTSP = NaN; 
        struct_feature_ABP(i).iLVET = NaN; % Left ventricular ejection time
        struct_feature_ABP(i).LVET = NaN; 
        struct_feature_ABP(i).iDT = NaN; % Diastolic time - Duration of the diastolic
        struct_feature_ABP(i).DT = NaN;
        struct_feature_ABP(i).iHP = NaN;% Heart period
        struct_feature_ABP(i).HP = NaN;
        struct_feature_ABP(i).DTF = NaN; % Diastolic time fraction
        %% Pressure Features
        struct_feature_ABP(i).SBP = NaN; % systolic blood pressure
        struct_feature_ABP(i).DBP = NaN; % Diastolic blood pressure
        struct_feature_ABP(i).ESBP = NaN; % End-systolic BP / Dicrotic BP
        struct_feature_ABP(i).PP = NaN; % Pulse Pressure
        struct_feature_ABP(i).MAP = NaN; % Mean arterial pressure
        struct_feature_ABP(i).MPP = NaN; % Mean pulse pressure
        struct_feature_ABP(i).MSBP = NaN; % Mean systolic blood pressure
        struct_feature_ABP(i).MDBP = NaN; % Mean diastolic blood pressure
        struct_feature_ABP(i).Ti = NaN; % Travel time of the reflected wave
        struct_feature_ABP(i).Pi = NaN; % Blood pressure at inflection point
        struct_feature_ABP(i).AP = NaN; % Augmented pressure
        struct_feature_ABP(i).AIx = NaN; % Augmentation index
        struct_feature_ABP(i).FF = NaN; %Form Factor
        %% Area
        struct_feature_ABP(i).SPTI = NaN; % Systolic pressure�time index (tension�time index)
        struct_feature_ABP(i).DPTI = NaN; % Diastolic pressure�time index
        struct_feature_ABP(i).SEVR = NaN; % Subendocardial viability ratio
        % Amplification Phenom.
        %Ampli = PSBP - CSBP; % Periferal BP - Central BP
        %Pulse pressure amplification
        %PPA = (PPP - CPP)/CPP;
    end


end