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
    sphere { m*<-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 1 }        
    sphere {  m*<0.14505211866252682,0.08233495146675299,3.444415262827075>, 1 }
    sphere {  m*<2.574428154821405,0.021121644361081382,-1.5740095639953384>, 1 }
    sphere {  m*<-1.7818955990777419,2.247561613393306,-1.318745803960125>, 1}
    sphere { m*<-1.51410837803991,-2.6401303290105913,-1.1291995187975523>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14505211866252682,0.08233495146675299,3.444415262827075>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5 }
    cylinder { m*<2.574428154821405,0.021121644361081382,-1.5740095639953384>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5}
    cylinder { m*<-1.7818955990777419,2.247561613393306,-1.318745803960125>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5 }
    cylinder {  m*<-1.51410837803991,-2.6401303290105913,-1.1291995187975523>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5}

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
    sphere { m*<-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 1 }        
    sphere {  m*<0.14505211866252682,0.08233495146675299,3.444415262827075>, 1 }
    sphere {  m*<2.574428154821405,0.021121644361081382,-1.5740095639953384>, 1 }
    sphere {  m*<-1.7818955990777419,2.247561613393306,-1.318745803960125>, 1}
    sphere { m*<-1.51410837803991,-2.6401303290105913,-1.1291995187975523>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.14505211866252682,0.08233495146675299,3.444415262827075>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5 }
    cylinder { m*<2.574428154821405,0.021121644361081382,-1.5740095639953384>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5}
    cylinder { m*<-1.7818955990777419,2.247561613393306,-1.318745803960125>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5 }
    cylinder {  m*<-1.51410837803991,-2.6401303290105913,-1.1291995187975523>, <-0.16028023918485185,-0.08091233102529274,-0.34480003854415525>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    