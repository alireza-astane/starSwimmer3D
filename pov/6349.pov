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
    sphere { m*<-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 1 }        
    sphere {  m*<0.09549121756239143,0.056317359158296415,8.969325432092234>, 1 }
    sphere {  m*<7.450842655562362,-0.03260291683606087,-5.610167857953121>, 1 }
    sphere {  m*<-4.539230846581144,3.5480572649440347,-2.535815950519906>, 1}
    sphere { m*<-2.6833031078569363,-3.1751102851914372,-1.5583751641648371>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09549121756239143,0.056317359158296415,8.969325432092234>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5 }
    cylinder { m*<7.450842655562362,-0.03260291683606087,-5.610167857953121>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5}
    cylinder { m*<-4.539230846581144,3.5480572649440347,-2.535815950519906>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5 }
    cylinder {  m*<-2.6833031078569363,-3.1751102851914372,-1.5583751641648371>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5}

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
    sphere { m*<-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 1 }        
    sphere {  m*<0.09549121756239143,0.056317359158296415,8.969325432092234>, 1 }
    sphere {  m*<7.450842655562362,-0.03260291683606087,-5.610167857953121>, 1 }
    sphere {  m*<-4.539230846581144,3.5480572649440347,-2.535815950519906>, 1}
    sphere { m*<-2.6833031078569363,-3.1751102851914372,-1.5583751641648371>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09549121756239143,0.056317359158296415,8.969325432092234>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5 }
    cylinder { m*<7.450842655562362,-0.03260291683606087,-5.610167857953121>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5}
    cylinder { m*<-4.539230846581144,3.5480572649440347,-2.535815950519906>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5 }
    cylinder {  m*<-2.6833031078569363,-3.1751102851914372,-1.5583751641648371>, <-1.3570030853544848,-0.5650180720424129,-0.9051781422320233>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    