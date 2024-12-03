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
    sphere { m*<-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 1 }        
    sphere {  m*<0.6384797353519661,-0.23078682182052934,9.238199310088042>, 1 }
    sphere {  m*<8.006266933674773,-0.5158790726127914,-5.3324781189858985>, 1 }
    sphere {  m*<-6.889696260014227,6.0072023010078635,-3.8416712158042925>, 1}
    sphere { m*<-2.173296700569057,-4.253556092988881,-1.255990649000827>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6384797353519661,-0.23078682182052934,9.238199310088042>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5 }
    cylinder { m*<8.006266933674773,-0.5158790726127914,-5.3324781189858985>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5}
    cylinder { m*<-6.889696260014227,6.0072023010078635,-3.8416712158042925>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5 }
    cylinder {  m*<-2.173296700569057,-4.253556092988881,-1.255990649000827>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5}

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
    sphere { m*<-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 1 }        
    sphere {  m*<0.6384797353519661,-0.23078682182052934,9.238199310088042>, 1 }
    sphere {  m*<8.006266933674773,-0.5158790726127914,-5.3324781189858985>, 1 }
    sphere {  m*<-6.889696260014227,6.0072023010078635,-3.8416712158042925>, 1}
    sphere { m*<-2.173296700569057,-4.253556092988881,-1.255990649000827>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6384797353519661,-0.23078682182052934,9.238199310088042>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5 }
    cylinder { m*<8.006266933674773,-0.5158790726127914,-5.3324781189858985>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5}
    cylinder { m*<-6.889696260014227,6.0072023010078635,-3.8416712158042925>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5 }
    cylinder {  m*<-2.173296700569057,-4.253556092988881,-1.255990649000827>, <-0.7806877588481969,-1.2207257357004473,-0.6110907869471145>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    