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
    sphere { m*<-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 1 }        
    sphere {  m*<-0.049293596329820416,0.2784546469906619,8.790251063168759>, 1 }
    sphere {  m*<6.820662623309132,0.10335721694865854,-5.452759240067767>, 1 }
    sphere {  m*<-3.1365758780101975,2.1479350824852146,-1.978942414926287>, 1}
    sphere { m*<-2.868788656972366,-2.739756859918683,-1.7893961297637166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.049293596329820416,0.2784546469906619,8.790251063168759>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5 }
    cylinder { m*<6.820662623309132,0.10335721694865854,-5.452759240067767>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5}
    cylinder { m*<-3.1365758780101975,2.1479350824852146,-1.978942414926287>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5 }
    cylinder {  m*<-2.868788656972366,-2.739756859918683,-1.7893961297637166>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5}

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
    sphere { m*<-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 1 }        
    sphere {  m*<-0.049293596329820416,0.2784546469906619,8.790251063168759>, 1 }
    sphere {  m*<6.820662623309132,0.10335721694865854,-5.452759240067767>, 1 }
    sphere {  m*<-3.1365758780101975,2.1479350824852146,-1.978942414926287>, 1}
    sphere { m*<-2.868788656972366,-2.739756859918683,-1.7893961297637166>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.049293596329820416,0.2784546469906619,8.790251063168759>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5 }
    cylinder { m*<6.820662623309132,0.10335721694865854,-5.452759240067767>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5}
    cylinder { m*<-3.1365758780101975,2.1479350824852146,-1.978942414926287>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5 }
    cylinder {  m*<-2.868788656972366,-2.739756859918683,-1.7893961297637166>, <-1.4634534755043784,-0.1813389741895884,-1.0985267941100565>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    