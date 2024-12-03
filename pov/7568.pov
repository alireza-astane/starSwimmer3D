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
    sphere { m*<-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 1 }        
    sphere {  m*<0.8808494404171722,0.2970470712323734,9.35043770284846>, 1 }
    sphere {  m*<8.248636638739976,0.01195482044011209,-5.220239726225468>, 1 }
    sphere {  m*<-6.647326554949021,6.535036194060754,-3.7294328230438607>, 1}
    sphere { m*<-3.3993325404075176,-6.92362275907742,-1.8237525815126838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8808494404171722,0.2970470712323734,9.35043770284846>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5 }
    cylinder { m*<8.248636638739976,0.01195482044011209,-5.220239726225468>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5}
    cylinder { m*<-6.647326554949021,6.535036194060754,-3.7294328230438607>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5 }
    cylinder {  m*<-3.3993325404075176,-6.92362275907742,-1.8237525815126838>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5}

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
    sphere { m*<-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 1 }        
    sphere {  m*<0.8808494404171722,0.2970470712323734,9.35043770284846>, 1 }
    sphere {  m*<8.248636638739976,0.01195482044011209,-5.220239726225468>, 1 }
    sphere {  m*<-6.647326554949021,6.535036194060754,-3.7294328230438607>, 1}
    sphere { m*<-3.3993325404075176,-6.92362275907742,-1.8237525815126838>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8808494404171722,0.2970470712323734,9.35043770284846>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5 }
    cylinder { m*<8.248636638739976,0.01195482044011209,-5.220239726225468>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5}
    cylinder { m*<-6.647326554949021,6.535036194060754,-3.7294328230438607>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5 }
    cylinder {  m*<-3.3993325404075176,-6.92362275907742,-1.8237525815126838>, <-0.5383180537829894,-0.6928918426475438,-0.49885239418668564>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    