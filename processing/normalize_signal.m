function nsig = normalize_signal(s)
    % Original from:
    % Slapničar, G.; Mlakar, N.; Luštrek, M. Blood Pressure Estimation from Photoplethysmogram UsingSpectro-Temporal Deep Neural Network. 19, 3420. doi:10.3390/s19153420.
    % Source:
    % https://github.com/gslapnicar/bp-estimation-mimic3/blob/master/cleaning_scripts/


    % Normalize the signal -> to an interval [0,1]
    nsig = (s - min(s))/(max(s)-min(s));
end