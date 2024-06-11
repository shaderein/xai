## BDD
- `func_levels`
    - Initial resutls. Used functional layer definition in Guoyang's paper. From F10 forward, it clamped the layers by the "detection route" determined by the scale of the proposed targets
- `skip_0`
    - 
- `same_layer`
    - Stopped using functional layers definition. Keep F10-F17 the same as the original layer in yolov5s.
- `same_layer_larger_font`
    - Same as previous. Just redraw the plots.
- (All results above use the old layer naming in the pickle file: F1=deepest F17=lowest)
- (Manually rename layers @24-03-29)
- `remove_empty_saliency`
    - Detected many zero correlation data points in the t-test bar plot. It's due to the unbalanced sample sizes between F1-13 and F14-17, as the later are not used when detecting large objects.
    - csv files used for downstream analysis in matlab and R. F1-13 only. Sorted layers

## MSCOCO

- No layers removed. All backbone layers. F1-F13 sorted. `H:\OneDrive - The University Of Hong Kong\mscoco\xai_saliency_maps_faster\fullgradcamraw`