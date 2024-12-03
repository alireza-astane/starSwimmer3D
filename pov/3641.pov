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
    sphere { m*<0.054632815930403866,0.31115390831526,-0.09639654836459915>, 1 }        
    sphere {  m*<0.2953679206720955,0.43986398649558534,2.891158222755951>, 1 }
    sphere {  m*<2.789341209936663,0.4131878837016344,-1.3256060738157847>, 1 }
    sphere {  m*<-1.5669825439624874,2.639627852733862,-1.07034231378057>, 1}
    sphere { m*<-2.430093094914906,-4.385864190653741,-1.536031061122991>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2953679206720955,0.43986398649558534,2.891158222755951>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5 }
    cylinder { m*<2.789341209936663,0.4131878837016344,-1.3256060738157847>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5}
    cylinder { m*<-1.5669825439624874,2.639627852733862,-1.07034231378057>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5 }
    cylinder {  m*<-2.430093094914906,-4.385864190653741,-1.536031061122991>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5}

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
    sphere { m*<0.054632815930403866,0.31115390831526,-0.09639654836459915>, 1 }        
    sphere {  m*<0.2953679206720955,0.43986398649558534,2.891158222755951>, 1 }
    sphere {  m*<2.789341209936663,0.4131878837016344,-1.3256060738157847>, 1 }
    sphere {  m*<-1.5669825439624874,2.639627852733862,-1.07034231378057>, 1}
    sphere { m*<-2.430093094914906,-4.385864190653741,-1.536031061122991>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2953679206720955,0.43986398649558534,2.891158222755951>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5 }
    cylinder { m*<2.789341209936663,0.4131878837016344,-1.3256060738157847>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5}
    cylinder { m*<-1.5669825439624874,2.639627852733862,-1.07034231378057>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5 }
    cylinder {  m*<-2.430093094914906,-4.385864190653741,-1.536031061122991>, <0.054632815930403866,0.31115390831526,-0.09639654836459915>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    