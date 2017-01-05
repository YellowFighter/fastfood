function [X_test,y_test, X_train, y_train] = load_music(path)

data = load(path);
y_music = data(:,1);
x_music = data(:,2:end);
y_music_scaled = zscore(y_music);
x_music_scaled = zscore(x_music);
% Data meant to be partitioned this way
y_train = y_music_scaled(1:463715,:);
X_train = x_music_scaled(1:463715,:);
y_test = y_music_scaled(463716:end,:);
X_test = x_music_scaled(463716:end,:);

end

