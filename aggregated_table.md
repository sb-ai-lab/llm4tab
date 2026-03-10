# 📊 Aggregated Experiments Results for "LLMTabBench: Evaluating LLMs on Tabular Data From Zero to Few Shots"

## 📋 Structure of the Results File

Below are the detailed results of all experiment launches for the article, organized into two main sections: by serialization (for LLM) and by shots (for LLM and baselines).

### File Organization

The file is divided into 3 main groups, corresponding to the three prompts used for dataset generation:

- **Group 1**: Task description with context (domain information, interpretation of the target column)

- **Group 2**: Task description without context (masked)

- **Group 3**: Presentation of crucial rules for target formation (only for LLM-generated datasets)

### Internal Structure

Each of the three main groups contains two subgroups, representing the two data sections:

📊 By serialization
🎯 By shots

Within each subgroup, the datasets are further organized into 9 categories:

8 categories — one for each knowledge domain
1 category — for MLP-synthetic datasets


# Group 1: Contextual Prompt

## Aggregated by Serialization

### Business Domain

\begin{table*}[ht]
\sisetup{
  separate-uncertainty = true,
  table-align-uncertainty = true,
  table-figures-uncertainty = 1,
}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}

\adjustbox{max width=\textwidth}{
\begin{tabular}{llcccccccccccccccccc}
\toprule
& & \multicolumn{18}{c}{\textbf{Models}} \\
\cmidrule(lr){3-20}
Dataset & Serialization & \multicolumn{6}{c}{Qwen3-8B} & \multicolumn{6}{c}{Qwen3-14B} & \multicolumn{6}{c}{GPT-4o-mini} \\
\cmidrule(lr){3-8}
\cmidrule(lr){9-14}
\cmidrule(lr){15-20}
 &  & shot 0 & shot 4 & shot 8 & shot 16 & shot 32 & shot 64 & shot 0 & shot 4 & shot 8 & shot 16 & shot 32 & shot 64 & shot 0 & shot 4 & shot 8 & shot 16 & shot 32 & shot 64 \\
\midrule
\multirow{5}{*}{callcenter} & feat\_val & \num{0.510 \pm 0.000} & \num{0.674 \pm 0.029} & \num{0.698 \pm 0.041} & \num{0.686 \pm 0.029} & \num{0.689 \pm 0.036} & \num{0.706 \pm 0.024} & \num{0.543 \pm 0.000} & \num{0.678 \pm 0.033} & \num{0.676 \pm 0.043} & \num{0.656 \pm 0.024} & \num{0.658 \pm 0.052} & \num{0.680 \pm 0.035} & \num{0.522 \pm 0.010} & \num{0.685 \pm 0.041} & \num{0.686 \pm 0.040} & \num{0.689 \pm 0.013} & \num{0.670 \pm 0.049} & \num{0.756 \pm 0.015} \\
 & feat\_val\_mask & \num{0.543 \pm 0.000} & \num{0.592 \pm 0.064} & \num{0.580 \pm 0.062} & \num{0.575 \pm 0.062} & \num{0.608 \pm 0.053} & \num{0.591 \pm 0.057} & \num{0.524 \pm 0.000} & \num{0.581 \pm 0.072} & \num{0.559 \pm 0.054} & \num{0.535 \pm 0.018} & \num{0.584 \pm 0.043} & \num{0.551 \pm 0.044} & \num{0.555 \pm 0.013} & \num{0.542 \pm 0.023} & \num{0.565 \pm 0.065} & \num{0.553 \pm 0.052} & \num{0.557 \pm 0.054} & -- \\
 & html & \num{0.503 \pm 0.000} & \num{0.670 \pm 0.075} & \num{0.707 \pm 0.034} & \num{0.689 \pm 0.065} & \num{0.657 \pm 0.032} & \num{0.683 \pm 0.032} & \num{0.539 \pm 0.000} & \num{0.711 \pm 0.029} & \num{0.725 \pm 0.030} & \num{0.710 \pm 0.036} & \num{0.706 \pm 0.031} & \num{0.707 \pm 0.020} & \num{0.510 \pm 0.010} & \num{0.685 \pm 0.015} & \num{0.697 \pm 0.045} & \num{0.683 \pm 0.035} & \num{0.669 \pm 0.023} & -- \\
 & markdown & \num{0.529 \pm 0.000} & \num{0.668 \pm 0.080} & \num{0.696 \pm 0.066} & \num{0.673 \pm 0.041} & \num{0.681 \pm 0.040} & \num{0.680 \pm 0.049} & \num{0.502 \pm 0.000} & \num{0.682 \pm 0.054} & \num{0.682 \pm 0.053} & \num{0.685 \pm 0.050} & \num{0.666 \pm 0.075} & \num{0.683 \pm 0.038} & \num{0.523 \pm 0.017} & \num{0.653 \pm 0.036} & \num{0.685 \pm 0.015} & \num{0.681 \pm 0.028} & \num{0.668 \pm 0.027} & -- \\
 & markdown\_mask & \num{0.505 \pm 0.002} & \num{0.638 \pm 0.055} & \num{0.654 \pm 0.053} & \num{0.654 \pm 0.036} & \num{0.644 \pm 0.036} & \num{0.664 \pm 0.036} & \num{0.568 \pm 0.000} & \num{0.637 \pm 0.079} & \num{0.640 \pm 0.066} & \num{0.632 \pm 0.032} & \num{0.588 \pm 0.049} & \num{0.588 \pm 0.041} & \num{0.538 \pm 0.013} & \num{0.595 \pm 0.048} & \num{0.608 \pm 0.046} & \num{0.588 \pm 0.061} & \num{0.554 \pm 0.038} & -- \\
\midrule
\multirow{5}{*}{hiring} & feat\_val & \num{0.564 \pm 0.000} & \num{0.572 \pm 0.021} & \num{0.580 \pm 0.020} & \num{0.586 \pm 0.023} & \num{0.573 \pm 0.014} & \num{0.563 \pm 0.028} & \num{0.553 \pm 0.000} & \num{0.556 \pm 0.021} & \num{0.562 \pm 0.026} & \num{0.554 \pm 0.020} & \num{0.553 \pm 0.020} & \num{0.548 \pm 0.018} & \num{0.573 \pm 0.003} & \num{0.581 \pm 0.019} & \num{0.582 \pm 0.020} & \num{0.583 \pm 0.020} & \num{0.596 \pm 0.023} & \num{0.605 \pm 0.022} \\
 & feat\_val\_mask & \num{0.596 \pm 0.000} & \num{0.628 \pm 0.008} & \num{0.596 \pm 0.022} & \num{0.604 \pm 0.055} & \num{0.580 \pm 0.033} & \num{0.609 \pm 0.054} & \num{0.641 \pm 0.000} & \num{0.603 \pm 0.038} & \num{0.594 \pm 0.032} & \num{0.602 \pm 0.038} & \num{0.578 \pm 0.045} & \num{0.589 \pm 0.048} & \num{0.605 \pm 0.012} & \num{0.603 \pm 0.026} & \num{0.592 \pm 0.023} & \num{0.613 \pm 0.031} & \num{0.619 \pm 0.034} & \num{0.605 \pm 0.024} \\
 & html & \num{0.558 \pm 0.000} & \num{0.556 \pm 0.012} & \num{0.581 \pm 0.019} & \num{0.586 \pm 0.031} & \num{0.569 \pm 0.017} & \num{0.589 \pm 0.020} & \num{0.521 \pm 0.000} & \num{0.536 \pm 0.012} & \num{0.538 \pm 0.032} & \num{0.548 \pm 0.019} & \num{0.538 \pm 0.021} & \num{0.545 \pm 0.013} & \num{0.552 \pm 0.007} & \num{0.581 \pm 0.014} & \num{0.584 \pm 0.017} & \num{0.595 \pm 0.019} & \num{0.580 \pm 0.015} & \num{0.582 \pm 0.016} \\
 & markdown & \num{0.583 \pm 0.000} & \num{0.571 \pm 0.009} & \num{0.578 \pm 0.044} & \num{0.597 \pm 0.027} & \num{0.570 \pm 0.020} & \num{0.549 \pm 0.015} & \num{0.529 \pm 0.000} & \num{0.539 \pm 0.006} & \num{0.541 \pm 0.021} & \num{0.563 \pm 0.017} & \num{0.555 \pm 0.019} & \num{0.555 \pm 0.013} & \num{0.578 \pm 0.005} & \num{0.565 \pm 0.011} & \num{0.572 \pm 0.006} & \num{0.575 \pm 0.011} & \num{0.566 \pm 0.022} & \num{0.574 \pm 0.019} \\
 & markdown\_mask & \num{0.623 \pm 0.007} & \num{0.608 \pm 0.030} & \num{0.579 \pm 0.046} & \num{0.621 \pm 0.022} & \num{0.576 \pm 0.036} & \num{0.584 \pm 0.045} & \num{0.624 \pm 0.000} & \num{0.614 \pm 0.039} & \num{0.599 \pm 0.034} & \num{0.624 \pm 0.016} & \num{0.612 \pm 0.035} & \num{0.598 \pm 0.034} & \num{0.617 \pm 0.005} & \num{0.577 \pm 0.022} & \num{0.565 \pm 0.030} & \num{0.579 \pm 0.015} & \num{0.568 \pm 0.013} & \num{0.578 \pm 0.022} \\
\midrule
\multirow{5}{*}{telco} & feat\_val & \num{0.619 \pm 0.000} & \num{0.723 \pm 0.049} & \num{0.737 \pm 0.064} & \num{0.752 \pm 0.008} & \num{0.727 \pm 0.031} & \num{0.747 \pm 0.016} & \num{0.827 \pm 0.000} & \num{0.787 \pm 0.043} & \num{0.782 \pm 0.032} & \num{0.802 \pm 0.020} & \num{0.804 \pm 0.012} & \num{0.814 \pm 0.004} & \num{0.807 \pm 0.001} & \num{0.790 \pm 0.021} & \num{0.788 \pm 0.022} & \num{0.803 \pm 0.010} & \num{0.730 \pm 0.051} & \num{0.756 \pm 0.036} \\
 & feat\_val\_mask & \num{0.515 \pm 0.000} & \num{0.536 \pm 0.041} & \num{0.627 \pm 0.101} & \num{0.660 \pm 0.031} & \num{0.705 \pm 0.036} & \num{0.725 \pm 0.037} & \num{0.676 \pm 0.000} & \num{0.751 \pm 0.036} & \num{0.776 \pm 0.033} & \num{0.784 \pm 0.016} & \num{0.804 \pm 0.005} & \num{0.809 \pm 0.017} & \num{0.576 \pm 0.005} & \num{0.681 \pm 0.066} & \num{0.741 \pm 0.063} & \num{0.789 \pm 0.012} & \num{0.740 \pm 0.054} & \num{0.764 \pm 0.030} \\
 & html & \num{0.580 \pm 0.001} & \num{0.665 \pm 0.053} & \num{0.736 \pm 0.067} & \num{0.748 \pm 0.016} & \num{0.762 \pm 0.030} & \num{0.765 \pm 0.012} & \num{0.816 \pm 0.000} & \num{0.790 \pm 0.029} & \num{0.766 \pm 0.036} & \num{0.760 \pm 0.043} & \num{0.779 \pm 0.009} & \num{0.782 \pm 0.006} & \num{0.741 \pm 0.004} & \num{0.808 \pm 0.015} & \num{0.810 \pm 0.008} & \num{0.816 \pm 0.007} & \num{0.759 \pm 0.028} & \num{0.752 \pm 0.044} \\
 & markdown & \num{0.504 \pm 0.001} & \num{0.641 \pm 0.082} & \num{0.670 \pm 0.064} & \num{0.701 \pm 0.025} & \num{0.735 \pm 0.051} & \num{0.713 \pm 0.045} & \num{0.761 \pm 0.000} & \num{0.801 \pm 0.026} & \num{0.801 \pm 0.018} & \num{0.765 \pm 0.035} & \num{0.787 \pm 0.010} & \num{0.806 \pm 0.014} & \num{0.773 \pm 0.003} & \num{0.812 \pm 0.017} & \num{0.808 \pm 0.015} & \num{0.816 \pm 0.009} & \num{0.760 \pm 0.032} & \num{0.770 \pm 0.011} \\
 & markdown\_mask & \num{0.547 \pm 0.009} & \num{0.554 \pm 0.054} & \num{0.602 \pm 0.060} & \num{0.651 \pm 0.063} & \num{0.649 \pm 0.047} & \num{0.591 \pm 0.071} & \num{0.639 \pm 0.000} & \num{0.759 \pm 0.030} & \num{0.742 \pm 0.020} & \num{0.725 \pm 0.017} & \num{0.739 \pm 0.011} & \num{0.753 \pm 0.011} & \num{0.761 \pm 0.003} & \num{0.730 \pm 0.034} & \num{0.783 \pm 0.036} & \num{0.780 \pm 0.028} & \num{0.739 \pm 0.048} & \num{0.734 \pm 0.030} \\
\midrule
\multirow{5}{*}{marketing} & feat\_val & \num{0.688 \pm 0.001} & \num{0.818 \pm 0.018} & \num{0.807 \pm 0.012} & \num{0.811 \pm 0.017} & \num{0.807 \pm 0.012} & \num{0.800 \pm 0.014} & \num{0.771 \pm 0.000} & \num{0.791 \pm 0.021} & \num{0.785 \pm 0.026} & \num{0.779 \pm 0.020} & \num{0.784 \pm 0.009} & \num{0.776 \pm 0.017} & \num{0.754 \pm 0.005} & \num{0.781 \pm 0.014} & \num{0.777 \pm 0.025} & \num{0.779 \pm 0.019} & \num{0.778 \pm 0.017} & \num{0.775 \pm 0.010} \\
 & feat\_val\_mask & \num{0.522 \pm 0.001} & \num{0.621 \pm 0.052} & \num{0.618 \pm 0.043} & \num{0.649 \pm 0.053} & \num{0.678 \pm 0.031} & \num{0.657 \pm 0.030} & \num{0.553 \pm 0.000} & \num{0.600 \pm 0.071} & \num{0.604 \pm 0.015} & \num{0.645 \pm 0.069} & \num{0.625 \pm 0.023} & \num{0.629 \pm 0.026} & \num{0.580 \pm 0.008} & \num{0.564 \pm 0.057} & \num{0.630 \pm 0.045} & \num{0.588 \pm 0.034} & \num{0.564 \pm 0.033} & \num{0.548 \pm 0.011} \\
 & html & \num{0.692 \pm 0.002} & \num{0.783 \pm 0.014} & \num{0.762 \pm 0.011} & \num{0.766 \pm 0.018} & \num{0.753 \pm 0.014} & \num{0.769 \pm 0.024} & \num{0.732 \pm 0.000} & \num{0.727 \pm 0.013} & \num{0.719 \pm 0.016} & \num{0.718 \pm 0.015} & \num{0.719 \pm 0.011} & \num{0.707 \pm 0.005} & \num{0.622 \pm 0.007} & \num{0.636 \pm 0.005} & \num{0.637 \pm 0.011} & \num{0.633 \pm 0.012} & \num{0.634 \pm 0.008} & \num{0.641 \pm 0.014} \\
 & markdown & \num{0.655 \pm 0.010} & \num{0.773 \pm 0.021} & \num{0.765 \pm 0.033} & \num{0.790 \pm 0.017} & \num{0.778 \pm 0.007} & \num{0.779 \pm 0.004} & \num{0.673 \pm 0.000} & \num{0.748 \pm 0.008} & \num{0.743 \pm 0.009} & \num{0.739 \pm 0.013} & \num{0.742 \pm 0.024} & \num{0.723 \pm 0.017} & \num{0.613 \pm 0.007} & \num{0.673 \pm 0.009} & \num{0.671 \pm 0.014} & \num{0.654 \pm 0.024} & \num{0.659 \pm 0.012} & \num{0.661 \pm 0.024} \\
 & markdown\_mask & \num{0.618 \pm 0.010} & \num{0.602 \pm 0.023} & \num{0.593 \pm 0.019} & \num{0.625 \pm 0.080} & \num{0.584 \pm 0.049} & \num{0.643 \pm 0.025} & \num{0.562 \pm 0.000} & \num{0.548 \pm 0.046} & \num{0.546 \pm 0.035} & \num{0.565 \pm 0.063} & \num{0.554 \pm 0.028} & \num{0.538 \pm 0.020} & \num{0.542 \pm 0.008} & \num{0.551 \pm 0.042} & \num{0.564 \pm 0.025} & \num{0.559 \pm 0.031} & \num{0.616 \pm 0.034} & \num{0.625 \pm 0.020} \\
\bottomrule
\end{tabular}
}

\caption{business - serializations}
\label{tab:serialization_results}
\end{table*}

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data

## Aggregated by Shots (with Baselines)

### Business Domain

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data

# Group 2: No-Context Prompt (Masked)

## Aggregated by Serialization

### Business Domain

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data

## Aggregated by Shots (with Baselines)

### Business Domain

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data

# Group 3: Decision Rules Prompt (for LLM-generated Data)

## Aggregated by Serialization

### Business Domain

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data

## Aggregated by Shots (with Baselines)

### Business Domain

### Education Domain

### Finance Domain

### Healthcare Domain

### Law Domain

### Natural Science Domain

### People & Society Domain

### Software & Engineering Domain

### MLP-generated Synthetic Data
