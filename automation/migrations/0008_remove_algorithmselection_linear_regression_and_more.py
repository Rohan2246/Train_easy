# Generated by Django 5.0.3 on 2024-04-29 05:05

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("automation", "0007_alter_dataset_name"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="algorithmselection",
            name="linear_Regression",
        ),
        migrations.RemoveField(
            model_name="algorithmselection",
            name="logistic_Regression",
        ),
        migrations.AddField(
            model_name="algorithmselection",
            name="knn",
            field=models.BooleanField(
                default=False, help_text="K-Nearest Neighbors models"
            ),
        ),
        migrations.AddField(
            model_name="algorithmselection",
            name="linear",
            field=models.BooleanField(default=False, help_text="Linear models"),
        ),
        migrations.AlterField(
            model_name="algorithmselection",
            name="decision_Tree",
            field=models.BooleanField(default=False, help_text="Tree based models"),
        ),
        migrations.AlterField(
            model_name="algorithmselection",
            name="naive_Bayes",
            field=models.BooleanField(default=False, help_text="Naive Bayes models"),
        ),
        migrations.AlterField(
            model_name="algorithmselection",
            name="random_Forest",
            field=models.BooleanField(default=False, help_text="Ensemble models"),
        ),
        migrations.AlterField(
            model_name="algorithmselection",
            name="support_Vector_Machines",
            field=models.BooleanField(default=False, help_text="SVM models"),
        ),
        migrations.AlterField(
            model_name="dataset",
            name="name",
            field=models.CharField(help_text="Automation Project Name", max_length=255),
        ),
        migrations.AlterField(
            model_name="preprocessing",
            name="pca",
            field=models.BooleanField(default=False, help_text="3 components for PCA"),
        ),
    ]
