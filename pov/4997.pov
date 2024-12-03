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
    sphere { m*<-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 1 }        
    sphere {  m*<0.5341647577465655,0.2903757282832376,8.273355033145096>, 1 }
    sphere {  m*<2.467546233102646,-0.03602324323329705,-2.900428473776017>, 1 }
    sphere {  m*<-1.888777520796501,2.1904167257989275,-2.645164713740803>, 1}
    sphere { m*<-1.6209902997586692,-2.69727521660497,-2.4556184285782305>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5341647577465655,0.2903757282832376,8.273355033145096>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5 }
    cylinder { m*<2.467546233102646,-0.03602324323329705,-2.900428473776017>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5}
    cylinder { m*<-1.888777520796501,2.1904167257989275,-2.645164713740803>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5 }
    cylinder {  m*<-1.6209902997586692,-2.69727521660497,-2.4556184285782305>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5}

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
    sphere { m*<-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 1 }        
    sphere {  m*<0.5341647577465655,0.2903757282832376,8.273355033145096>, 1 }
    sphere {  m*<2.467546233102646,-0.03602324323329705,-2.900428473776017>, 1 }
    sphere {  m*<-1.888777520796501,2.1904167257989275,-2.645164713740803>, 1}
    sphere { m*<-1.6209902997586692,-2.69727521660497,-2.4556184285782305>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5341647577465655,0.2903757282832376,8.273355033145096>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5 }
    cylinder { m*<2.467546233102646,-0.03602324323329705,-2.900428473776017>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5}
    cylinder { m*<-1.888777520796501,2.1904167257989275,-2.645164713740803>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5 }
    cylinder {  m*<-1.6209902997586692,-2.69727521660497,-2.4556184285782305>, <-0.26716216090361106,-0.13805721861967124,-1.6712189483248352>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    