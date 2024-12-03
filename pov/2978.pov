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
    sphere { m*<0.5263075224630063,1.157244568980963,0.17705893949649115>, 1 }        
    sphere {  m*<0.7673567274099631,1.2706430690535409,3.1652038516718344>, 1 }
    sphere {  m*<3.260603916472499,1.2706430690535402,-1.052078356818782>, 1 }
    sphere {  m*<-1.1793125271509037,3.602156740558251,-0.8314031658224423>, 1}
    sphere { m*<-3.972249474580385,-7.368207017989949,-2.48211598738464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7673567274099631,1.2706430690535409,3.1652038516718344>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5 }
    cylinder { m*<3.260603916472499,1.2706430690535402,-1.052078356818782>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5}
    cylinder { m*<-1.1793125271509037,3.602156740558251,-0.8314031658224423>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5 }
    cylinder {  m*<-3.972249474580385,-7.368207017989949,-2.48211598738464>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5}

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
    sphere { m*<0.5263075224630063,1.157244568980963,0.17705893949649115>, 1 }        
    sphere {  m*<0.7673567274099631,1.2706430690535409,3.1652038516718344>, 1 }
    sphere {  m*<3.260603916472499,1.2706430690535402,-1.052078356818782>, 1 }
    sphere {  m*<-1.1793125271509037,3.602156740558251,-0.8314031658224423>, 1}
    sphere { m*<-3.972249474580385,-7.368207017989949,-2.48211598738464>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7673567274099631,1.2706430690535409,3.1652038516718344>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5 }
    cylinder { m*<3.260603916472499,1.2706430690535402,-1.052078356818782>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5}
    cylinder { m*<-1.1793125271509037,3.602156740558251,-0.8314031658224423>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5 }
    cylinder {  m*<-3.972249474580385,-7.368207017989949,-2.48211598738464>, <0.5263075224630063,1.157244568980963,0.17705893949649115>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    