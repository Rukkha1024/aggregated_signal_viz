<log>
(module) alice@DESKTOP-2HCKVI9:/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz$ python 
main.py
[filter] Warning: column 'mixed' not found in metadata, skipping this filter    
[filter] Warning: column 'age_group' not found in metadata, skipping this filter
Traceback (most recent call last):
  File "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/main.py", line 15, in <module> 
    main()
  File "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/main.py", line 11, in main     
    visualizer.run(modes=args.modes, signal_groups=args.groups, sample=args.sample)
  File "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/script/visualizer.py", line 1296, in run
    tasks.extend(self._build_plot_tasks(resampled, group_name, mode_name, mode_cfg))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/script/visualizer.py", line 1566, in _build_plot_tasks
    markers_by_key[key] = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/script/visualizer.py", line 2094, in _collect_markers
    col = filter_cfg["column"]
          ~~~~~~~~~~^^^^^^^^^^
KeyError: 'column'
</log>

문제 발생. 