function secs = stamp_to_seconds(stamp)

    secs = double(stamp.Sec) + double(stamp.Nsec)*1E-9;

end 