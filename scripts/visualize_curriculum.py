#!/usr/bin/env python3
"""
Visualize curriculum stages for multi-layer NPT training.
This helps understand how the curriculum progresses during training.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_curriculum(curriculum_str):
    """Parse curriculum string into stages."""
    stages = []
    cumulative_steps = 0

    for stage_str in curriculum_str.split(','):
        parts = stage_str.strip().split(':')
        stage_name = parts[0]
        steps = int(parts[1])
        mixing_ratio = float(parts[2]) if len(parts) > 2 else 0.0

        stages.append({
            'name': stage_name,
            'start': cumulative_steps,
            'end': cumulative_steps + steps,
            'mixing_ratio': mixing_ratio
        })
        cumulative_steps += steps

    return stages, cumulative_steps


def visualize_curriculum(curriculum_str, output_file=None):
    """Create visualization of curriculum stages."""
    stages, total_steps = parse_curriculum(curriculum_str)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Colors for different stages
    colors = {
        'teacher': '#4CAF50',  # Green
        'mixed': '#FFC107',    # Amber
        'student': '#2196F3'   # Blue
    }

    # Plot 1: Stage progression
    for stage in stages:
        color = colors.get(stage['name'], '#999999')
        ax1.barh(0, stage['end'] - stage['start'],
                left=stage['start'], height=0.8,
                color=color, alpha=0.7,
                edgecolor='black', linewidth=2)

        # Add stage label
        mid_point = (stage['start'] + stage['end']) / 2
        ax1.text(mid_point, 0,
                f"{stage['name']}\n({stage['end']-stage['start']} steps)",
                ha='center', va='center', fontweight='bold')

    ax1.set_ylim(-0.5, 0.5)
    ax1.set_ylabel('Curriculum Stage', fontsize=12)
    ax1.set_title('Curriculum Stage Progression', fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3)

    # Plot 2: Mixing ratio over time
    steps = np.arange(0, total_steps)
    mixing_ratios = np.zeros(total_steps)
    stage_indices = np.zeros(total_steps)

    for i, step in enumerate(steps):
        for j, stage in enumerate(stages):
            if stage['start'] <= step < stage['end']:
                if stage['name'] == 'teacher':
                    mixing_ratios[i] = 0.0
                    stage_indices[i] = 0
                elif stage['name'] == 'mixed':
                    mixing_ratios[i] = stage['mixing_ratio']
                    stage_indices[i] = 1
                elif stage['name'] == 'student':
                    mixing_ratios[i] = 1.0
                    stage_indices[i] = 2
                break

    # Plot mixing ratio
    ax2.plot(steps, mixing_ratios, linewidth=2, color='#E91E63')
    ax2.fill_between(steps, 0, mixing_ratios, alpha=0.3, color='#E91E63')

    # Add stage boundaries
    for stage in stages[1:]:
        ax2.axvline(x=stage['start'], color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Student Input Ratio', fontsize=12)
    ax2.set_title('Attention Input Mixing Ratio (0=Teacher, 1=Student)', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['teacher'], alpha=0.7, label='Teacher (100% teacher inputs)'),
        Patch(facecolor=colors['mixed'], alpha=0.7, label='Mixed (gradual transition)'),
        Patch(facecolor=colors['student'], alpha=0.7, label='Student (100% student inputs)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()

    # Print summary
    print("\nCurriculum Summary:")
    print("=" * 60)
    for i, stage in enumerate(stages):
        duration = stage['end'] - stage['start']
        percentage = (duration / total_steps) * 100
        print(f"Stage {i+1}: {stage['name']:<10} | Steps {stage['start']:6d}-{stage['end']:6d} "
              f"({duration:6d} steps, {percentage:5.1f}%) | Mixing: {stage['mixing_ratio']:.1f}")
    print(f"Total training steps: {total_steps}")


def main():
    parser = argparse.ArgumentParser(description="Visualize curriculum stages")
    parser.add_argument(
        "--curriculum",
        type=str,
        default="teacher:5000,mixed:5000:0.3,mixed:5000:0.7,student:15000",
        help="Curriculum stages string"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for visualization (PNG)"
    )

    args = parser.parse_args()

    print(f"Visualizing curriculum: {args.curriculum}")
    visualize_curriculum(args.curriculum, args.output)


if __name__ == "__main__":
    main()