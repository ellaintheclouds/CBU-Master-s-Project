% Script to create sttc adjacency matrix from organoid recordings
% Created by Alex Dunn, Cambridge, 2024
% Updated by Elle Richardson, Cambridge, 2025

base_path = 'C:/Users/er05/OneDrive - University of Cambridge/University of Cambridge/Basic and Translational Neuroscience/Research Project/Data/';
lag = 0.01;  % STTC lag
Hz = 20000;  % Sampling frequency
parallel = 0;

% Recursively search for all .h5 files
all_h5_files = dir(fullfile(base_path, '**', '*.h5'));

% Filter to include only those within a "Network" folder or subfolder
valid_h5_files = {};
for i = 1:length(all_h5_files)
    file_path = fullfile(all_h5_files(i).folder, all_h5_files(i).name);
    
    % Break path into folder parts and check if "Network" is one
    if any(strcmp(split(fileparts(file_path), filesep), 'Network'))
        valid_h5_files{end+1} = file_path;  % Add to list
    end
end

% Remove duplicates, just in case
valid_h5_files = unique(valid_h5_files);

% Print the number of Network .h5 files found
fprintf('Found %d valid .h5 files for processing:\n', numel(valid_h5_files));
for i = 1:numel(valid_h5_files)
    fprintf('  %s\n', valid_h5_files{i});
end

% Proceed with STTC processing loop
for i = 1:length(valid_h5_files)
    try
        file = valid_h5_files{i};
        [~, name, ~] = fileparts(file);
        filename = ['H_d_s_dt', num2str(lag*1000), '.mat'];
        fprintf('\nProcessing file %d of %d:\n  %s\n', i, numel(valid_h5_files), file);

        % Load and preprocess spike data
        data = h5read(file, '/proc0/spikeTimes');
        rec_dur = h5read(file, '/assay/inputs/record_time');
        coords = h5read(file, '/mapping');

        missing_channels = setdiff(unique(data.channel), coords.channel);
        active_channel_idx = find(ismember(coords.channel, unique(data.channel)));
        x = coords.x(active_channel_idx);
        y = coords.y(active_channel_idx);
        coord_channel_idx = find(ismember(data.channel, coords.channel));

        spiketimes = data.frameno(coord_channel_idx);
        spikechannels = data.channel(coord_channel_idx);

        adjM = sttc_times_fcn(double(spiketimes), double(spikechannels), lag, Hz, parallel, ...
            double([min(data.frameno)/Hz, max(data.frameno)/Hz]));
        
        % Extract and parse components from the file path
        path_parts = split(file, filesep); % filePath is the full path to .h5 file

        % Identify cell line
        cell_types = {'fiaj', 'scti003a', 'hehd', 'bioni', 'pahc4'};
        cell_line = 'unknown';
        for ct = cell_types
            match = contains(lower(path_parts), ct{1});
            if any(match)
                cell_line = ct{1};
                break;
            end
        end

        % Start date (gYYYYMMDD)
        start_date = 'unknownstart';
        for i = 1:length(path_parts)
            token = regexp(path_parts{i}, 'g(\d{8})', 'tokens');
            if ~isempty(token)
                start_date = token{1}{1};
                break;
            end
        end

        % Record date (YYMMDD → YYYYMMDD)
        record_date = 'unknownrec';
        for i = 1:length(path_parts)
            if ~isempty(regexp(path_parts{i}, '^\d{6}$', 'once'))
                record_date = ['20', path_parts{i}];
                break;
            end
        end

        % MEA chip ID (5-digit number)
        mea_id = 'unknownmea';
        for i = 1:length(path_parts)
            if ~isempty(regexp(path_parts{i}, '^\d{5}$', 'once'))
                mea_id = path_parts{i};
                break;
            end
        end

        % Slice number (6-digit number with leading zeros removed)
        slice_id = 'unknownslice';
        for i = length(path_parts):-1:1  % Search backwards
            if ~isempty(regexp(path_parts{i}, '^\d{6}$', 'once'))
                slice_id = regexprep(path_parts{i}, '^0+', '');  % Remove leading zeros
                break;
            end
        end

        % Construct final filename
        filename = sprintf('%s_%s_%s_mea%s_s%s.mat', cell_line, start_date, record_date, mea_id, slice_id);

        % Define the new save directory
        save_path = 'V:/er05/H5 Analysis/Organoid Data';

        % Ensure the directory exists
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end

        % Save result to same folder as original file
        save(fullfile(save_path, filename), 'adjM', 'data', 'coords', 'x', 'y', 'spiketimes', 'spikechannels', 'rec_dur', 'Hz', 'lag');
        fprintf('Saved: %s\n', save_path);

    catch ME
        fprintf('ERROR processing %s\nReason: %s\n', file, ME.message);
    end
end
