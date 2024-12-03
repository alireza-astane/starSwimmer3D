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
    sphere { m*<-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 1 }        
    sphere {  m*<0.9329998935529308,0.4106205753785064,9.374587928389692>, 1 }
    sphere {  m*<8.300787091875732,0.1255283245862442,-5.196089500684242>, 1 }
    sphere {  m*<-6.595176101813267,6.64860969820688,-3.7052825975026353>, 1}
    sphere { m*<-3.6425839840004244,-7.4533769064773185,-1.936399296407094>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9329998935529308,0.4106205753785064,9.374587928389692>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5 }
    cylinder { m*<8.300787091875732,0.1255283245862442,-5.196089500684242>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5}
    cylinder { m*<-6.595176101813267,6.64860969820688,-3.7052825975026353>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5 }
    cylinder {  m*<-3.6425839840004244,-7.4533769064773185,-1.936399296407094>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5}

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
    sphere { m*<-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 1 }        
    sphere {  m*<0.9329998935529308,0.4106205753785064,9.374587928389692>, 1 }
    sphere {  m*<8.300787091875732,0.1255283245862442,-5.196089500684242>, 1 }
    sphere {  m*<-6.595176101813267,6.64860969820688,-3.7052825975026353>, 1}
    sphere { m*<-3.6425839840004244,-7.4533769064773185,-1.936399296407094>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9329998935529308,0.4106205753785064,9.374587928389692>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5 }
    cylinder { m*<8.300787091875732,0.1255283245862442,-5.196089500684242>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5}
    cylinder { m*<-6.595176101813267,6.64860969820688,-3.7052825975026353>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5 }
    cylinder {  m*<-3.6425839840004244,-7.4533769064773185,-1.936399296407094>, <-0.4861676006472312,-0.579318338501411,-0.47470216864545794>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    