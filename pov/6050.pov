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
    sphere { m*<-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 1 }        
    sphere {  m*<-0.09650439701274172,0.21889970516219492,8.871465942554655>, 1 }
    sphere {  m*<7.258847040987226,0.12997942916783728,-5.708027347490708>, 1 }
    sphere {  m*<-3.456134920296995,2.3562013551926766,-1.9820871633236443>, 1}
    sphere { m*<-2.953131210551657,-2.8057899787229412,-1.6967661994595375>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.09650439701274172,0.21889970516219492,8.871465942554655>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5 }
    cylinder { m*<7.258847040987226,0.12997942916783728,-5.708027347490708>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5}
    cylinder { m*<-3.456134920296995,2.3562013551926766,-1.9820871633236443>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5 }
    cylinder {  m*<-2.953131210551657,-2.8057899787229412,-1.6967661994595375>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5}

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
    sphere { m*<-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 1 }        
    sphere {  m*<-0.09650439701274172,0.21889970516219492,8.871465942554655>, 1 }
    sphere {  m*<7.258847040987226,0.12997942916783728,-5.708027347490708>, 1 }
    sphere {  m*<-3.456134920296995,2.3562013551926766,-1.9820871633236443>, 1}
    sphere { m*<-2.953131210551657,-2.8057899787229412,-1.6967661994595375>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.09650439701274172,0.21889970516219492,8.871465942554655>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5 }
    cylinder { m*<7.258847040987226,0.12997942916783728,-5.708027347490708>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5}
    cylinder { m*<-3.456134920296995,2.3562013551926766,-1.9820871633236443>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5 }
    cylinder {  m*<-2.953131210551657,-2.8057899787229412,-1.6967661994595375>, <-1.5611796979684356,-0.23881242362724725,-1.0101831035615505>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    