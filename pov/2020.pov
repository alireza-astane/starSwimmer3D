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
    sphere { m*<1.263191589314972,0.02845233786136132,0.6127508733480922>, 1 }        
    sphere {  m*<1.5074382111322482,0.030508742905678576,3.602790808624775>, 1 }
    sphere {  m*<4.000685400194785,0.030508742905678583,-0.6144913998658412>, 1 }
    sphere {  m*<-3.641769449880598,8.066619939021098,-2.2873910003824713>, 1}
    sphere { m*<-3.697809377619109,-8.143353415498638,-2.3198348111614155>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5074382111322482,0.030508742905678576,3.602790808624775>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5 }
    cylinder { m*<4.000685400194785,0.030508742905678583,-0.6144913998658412>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5}
    cylinder { m*<-3.641769449880598,8.066619939021098,-2.2873910003824713>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5 }
    cylinder {  m*<-3.697809377619109,-8.143353415498638,-2.3198348111614155>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5}

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
    sphere { m*<1.263191589314972,0.02845233786136132,0.6127508733480922>, 1 }        
    sphere {  m*<1.5074382111322482,0.030508742905678576,3.602790808624775>, 1 }
    sphere {  m*<4.000685400194785,0.030508742905678583,-0.6144913998658412>, 1 }
    sphere {  m*<-3.641769449880598,8.066619939021098,-2.2873910003824713>, 1}
    sphere { m*<-3.697809377619109,-8.143353415498638,-2.3198348111614155>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5074382111322482,0.030508742905678576,3.602790808624775>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5 }
    cylinder { m*<4.000685400194785,0.030508742905678583,-0.6144913998658412>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5}
    cylinder { m*<-3.641769449880598,8.066619939021098,-2.2873910003824713>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5 }
    cylinder {  m*<-3.697809377619109,-8.143353415498638,-2.3198348111614155>, <1.263191589314972,0.02845233786136132,0.6127508733480922>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    