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
    sphere { m*<-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 1 }        
    sphere {  m*<0.4084627465982769,0.28804801622417037,8.390937791403237>, 1 }
    sphere {  m*<3.7288906299602154,0.006961439974359468,-3.5728008618510474>, 1 }
    sphere {  m*<-2.203559024187033,2.1788279690399914,-2.4996423457317065>, 1}
    sphere { m*<-1.9357718031492015,-2.708863973363906,-2.310096060569136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4084627465982769,0.28804801622417037,8.390937791403237>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5 }
    cylinder { m*<3.7288906299602154,0.006961439974359468,-3.5728008618510474>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5}
    cylinder { m*<-2.203559024187033,2.1788279690399914,-2.4996423457317065>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5 }
    cylinder {  m*<-1.9357718031492015,-2.708863973363906,-2.310096060569136>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5}

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
    sphere { m*<-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 1 }        
    sphere {  m*<0.4084627465982769,0.28804801622417037,8.390937791403237>, 1 }
    sphere {  m*<3.7288906299602154,0.006961439974359468,-3.5728008618510474>, 1 }
    sphere {  m*<-2.203559024187033,2.1788279690399914,-2.4996423457317065>, 1}
    sphere { m*<-1.9357718031492015,-2.708863973363906,-2.310096060569136>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4084627465982769,0.28804801622417037,8.390937791403237>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5 }
    cylinder { m*<3.7288906299602154,0.006961439974359468,-3.5728008618510474>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5}
    cylinder { m*<-2.203559024187033,2.1788279690399914,-2.4996423457317065>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5 }
    cylinder {  m*<-1.9357718031492015,-2.708863973363906,-2.310096060569136>, <-0.5668423231754969,-0.14982725386384232,-1.551740347881422>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    