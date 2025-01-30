def write_CSV(df, output_path):

    df.to_csv(output_path, index=False)
    print(f"File ${output_path} written")

    return output_path