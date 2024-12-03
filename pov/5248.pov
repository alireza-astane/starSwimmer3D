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
    sphere { m*<-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 1 }        
    sphere {  m*<0.40902406166532873,0.2880599379834318,8.390452228240012>, 1 }
    sphere {  m*<3.7239821833839732,0.006798543577196309,-3.570071217892343>, 1 }
    sphere {  m*<-2.202219671089621,2.1788754664525944,-2.5003083552216836>, 1}
    sphere { m*<-1.9344324500517895,-2.708816475951303,-2.3107620700591136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40902406166532873,0.2880599379834318,8.390452228240012>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5 }
    cylinder { m*<3.7239821833839732,0.006798543577196309,-3.570071217892343>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5}
    cylinder { m*<-2.202219671089621,2.1788754664525944,-2.5003083552216836>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5 }
    cylinder {  m*<-1.9344324500517895,-2.708816475951303,-2.3107620700591136>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5}

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
    sphere { m*<-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 1 }        
    sphere {  m*<0.40902406166532873,0.2880599379834318,8.390452228240012>, 1 }
    sphere {  m*<3.7239821833839732,0.006798543577196309,-3.570071217892343>, 1 }
    sphere {  m*<-2.202219671089621,2.1788754664525944,-2.5003083552216836>, 1}
    sphere { m*<-1.9344324500517895,-2.708816475951303,-2.3107620700591136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40902406166532873,0.2880599379834318,8.390452228240012>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5 }
    cylinder { m*<3.7239821833839732,0.006798543577196309,-3.570071217892343>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5}
    cylinder { m*<-2.202219671089621,2.1788754664525944,-2.5003083552216836>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5 }
    cylinder {  m*<-1.9344324500517895,-2.708816475951303,-2.3107620700591136>, <-0.5655646683084987,-0.1497789327058264,-1.5522978036376058>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    