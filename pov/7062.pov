#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 1 }        
    sphere {  m*<0.6410826501127319,-0.22511818171103082,9.239404687505825>, 1 }
    sphere {  m*<8.00886984843554,-0.5102104325032926,-5.331272741568114>, 1 }
    sphere {  m*<-6.8870933452534615,6.012870941117362,-3.840465838386508>, 1}
    sphere { m*<-2.1879016632876307,-4.2853628500568215,-1.2627540253948266>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6410826501127319,-0.22511818171103082,9.239404687505825>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5 }
    cylinder { m*<8.00886984843554,-0.5102104325032926,-5.331272741568114>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5}
    cylinder { m*<-6.8870933452534615,6.012870941117362,-3.840465838386508>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5 }
    cylinder {  m*<-2.1879016632876307,-4.2853628500568215,-1.2627540253948266>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 1 }        
    sphere {  m*<0.6410826501127319,-0.22511818171103082,9.239404687505825>, 1 }
    sphere {  m*<8.00886984843554,-0.5102104325032926,-5.331272741568114>, 1 }
    sphere {  m*<-6.8870933452534615,6.012870941117362,-3.840465838386508>, 1}
    sphere { m*<-2.1879016632876307,-4.2853628500568215,-1.2627540253948266>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6410826501127319,-0.22511818171103082,9.239404687505825>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5 }
    cylinder { m*<8.00886984843554,-0.5102104325032926,-5.331272741568114>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5}
    cylinder { m*<-6.8870933452534615,6.012870941117362,-3.840465838386508>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5 }
    cylinder {  m*<-2.1879016632876307,-4.2853628500568215,-1.2627540253948266>, <-0.7780848440874308,-1.2150570955909488,-0.6098854095293297>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    