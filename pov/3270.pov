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
    sphere { m*<0.31023330031136076,0.7943299785620441,0.05169675930626713>, 1 }        
    sphere {  m*<0.5509684050530524,0.9230400567423697,3.0392515304268177>, 1 }
    sphere {  m*<3.044941694317617,0.8963639539484185,-1.1775127661449152>, 1 }
    sphere {  m*<-1.3113820595815295,3.122803922980646,-0.9222490061097016>, 1}
    sphere { m*<-3.3464565027361526,-6.118117833996238,-2.06696623685815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5509684050530524,0.9230400567423697,3.0392515304268177>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5 }
    cylinder { m*<3.044941694317617,0.8963639539484185,-1.1775127661449152>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5}
    cylinder { m*<-1.3113820595815295,3.122803922980646,-0.9222490061097016>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5 }
    cylinder {  m*<-3.3464565027361526,-6.118117833996238,-2.06696623685815>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5}

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
    sphere { m*<0.31023330031136076,0.7943299785620441,0.05169675930626713>, 1 }        
    sphere {  m*<0.5509684050530524,0.9230400567423697,3.0392515304268177>, 1 }
    sphere {  m*<3.044941694317617,0.8963639539484185,-1.1775127661449152>, 1 }
    sphere {  m*<-1.3113820595815295,3.122803922980646,-0.9222490061097016>, 1}
    sphere { m*<-3.3464565027361526,-6.118117833996238,-2.06696623685815>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5509684050530524,0.9230400567423697,3.0392515304268177>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5 }
    cylinder { m*<3.044941694317617,0.8963639539484185,-1.1775127661449152>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5}
    cylinder { m*<-1.3113820595815295,3.122803922980646,-0.9222490061097016>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5 }
    cylinder {  m*<-3.3464565027361526,-6.118117833996238,-2.06696623685815>, <0.31023330031136076,0.7943299785620441,0.05169675930626713>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    