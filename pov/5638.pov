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
    sphere { m*<-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 1 }        
    sphere {  m*<0.15868897219536832,0.2828393189959865,8.609492025102869>, 1 }
    sphere {  m*<5.581749913171109,0.06605859074001863,-4.665087919065711>, 1 }
    sphere {  m*<-2.747293385534405,2.160342592236337,-2.2086549195882963>, 1}
    sphere { m*<-2.4795061644965735,-2.7273493501675605,-2.019108634425726>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15868897219536832,0.2828393189959865,8.609492025102869>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5 }
    cylinder { m*<5.581749913171109,0.06605859074001863,-4.665087919065711>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5}
    cylinder { m*<-2.747293385534405,2.160342592236337,-2.2086549195882963>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5 }
    cylinder {  m*<-2.4795061644965735,-2.7273493501675605,-2.019108634425726>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5}

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
    sphere { m*<-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 1 }        
    sphere {  m*<0.15868897219536832,0.2828393189959865,8.609492025102869>, 1 }
    sphere {  m*<5.581749913171109,0.06605859074001863,-4.665087919065711>, 1 }
    sphere {  m*<-2.747293385534405,2.160342592236337,-2.2086549195882963>, 1}
    sphere { m*<-2.4795061644965735,-2.7273493501675605,-2.019108634425726>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15868897219536832,0.2828393189959865,8.609492025102869>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5 }
    cylinder { m*<5.581749913171109,0.06605859074001863,-4.665087919065711>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5}
    cylinder { m*<-2.747293385534405,2.160342592236337,-2.2086549195882963>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5 }
    cylinder {  m*<-2.4795061644965735,-2.7273493501675605,-2.019108634425726>, <-1.0877546819308357,-0.16866796308195653,-1.3022157971261703>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    