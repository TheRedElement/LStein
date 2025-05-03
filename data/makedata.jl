


function binning(x, y, n)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}
    
    intervsize = Int(floor(length(x)/n))
    
    x_b = []
    x_s = []
    y_b = []
    y_s = []

    for i in 1:n
        # println(((i-1)*intervsize+1):(i*intervsize+1))
        push!(x_b, nanmean(x[((i-1)*intervsize+1):(i*intervsize+1)]))
        push!(x_s, nanstd(x[((i-1)*intervsize+1):(i*intervsize+1)]))
        push!(y_b, nanmean(y[((i-1)*intervsize+1):(i*intervsize+1)]))
        push!(y_s, nanstd(y[((i-1)*intervsize+1):(i*intervsize+1)]))
    end

    return x_b, y_b, x_s, y_s
end


begin   #ogle
    ttp = "rrc"
    cls = "RRLYR"
    tid = "00015"
    p1 = 0.3557403
    # p2 = 0.4675207

    dfi = DataFrame(CSV.File("/home/lukas/github/LVisP/data/temp/OGLE-LMC-$(cls)-$(tid).dat", header=0))
    dfv = DataFrame(CSV.File("/home/lukas/github/LVisP/data/temp/OGLE-LMC-$(cls)-$(tid).dat-V", header=0))

    dfi[!,:flt_nm] .= 810
    dfv[!,:flt_nm] .= 550

    df = vcat(dfi, dfv)


    rename!(df, [:hjd_2450000, :mag, :mag_e, :flt_nm])
    transform!(df,
        :hjd_2450000 => (t -> mod.(t ./ p1,1)) => :ph,
        # :hjd_2450000 => (t -> mod.(t ./ p1,1)) => :ph1,
        # :hjd_2450000 => (t -> mod.(t ./ p2,1)) => :ph2,
    )

    select!(df, :flt_nm, :ph, :mag, :mag_e, :hjd_2450000)


    sort!(df, :ph)
    df[!,:processing] .= "raw"

    begin #binning
        for pb in unique(df[!,:flt_nm])
            df2b = subset(df, :flt_nm => p -> p .== pb)
            x_b, y_b, x_s, y_s = binning(df2b[!,:ph], df2b[!,:mag], 40)
            df_b = DataFrame(:flt_nm=>pb, :ph=>x_b, :mag=>y_b, :mag_e=>y_s, :processing=>"bng",)

            for n in names(df)
                if ~in(n, names(df_b))
                    df_b[!,n] .= NaN
                end
            end
            p = scatter(df2b[!,:ph], df2b[!,:mag])
            scatter!(p, df_b[!,:ph], df_b[!,:mag]; yerr=df_b[!,:mag_e])
            display(p)

            df = vcat(df, df_b)
        end
        # display(df)

    end
    
    #saving
    CSV.write(joinpath(@__DIR__, "../data/$(tid)_$(ttp)_ogle.csv"), df)


end
