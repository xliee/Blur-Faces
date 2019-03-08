namespace Blur
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.imgCamUser = new System.Windows.Forms.PictureBox();
            this.button1 = new System.Windows.Forms.Button();
            this.testimg = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.imgCamUser)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.testimg)).BeginInit();
            this.SuspendLayout();
            // 
            // imgCamUser
            // 
            this.imgCamUser.Location = new System.Drawing.Point(101, 35);
            this.imgCamUser.Name = "imgCamUser";
            this.imgCamUser.Size = new System.Drawing.Size(531, 378);
            this.imgCamUser.TabIndex = 0;
            this.imgCamUser.TabStop = false;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(668, 35);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 1;
            this.button1.Text = "button1";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // testimg
            // 
            this.testimg.Location = new System.Drawing.Point(668, 246);
            this.testimg.Name = "testimg";
            this.testimg.Size = new System.Drawing.Size(296, 167);
            this.testimg.TabIndex = 2;
            this.testimg.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(976, 450);
            this.Controls.Add(this.testimg);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.imgCamUser);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.imgCamUser)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.testimg)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox imgCamUser;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PictureBox testimg;
    }
}

