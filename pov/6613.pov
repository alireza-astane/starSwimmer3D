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
    sphere { m*<-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 1 }        
    sphere {  m*<0.293813670191454,-0.10818066720577704,9.0703886554506>, 1 }
    sphere {  m*<7.649165108191419,-0.19710094320013397,-5.509104634594742>, 1 }
    sphere {  m*<-5.515026912078784,4.545324619746131,-3.034220506926549>, 1}
    sphere { m*<-2.421157365789743,-3.4978359790755538,-1.4241447839691497>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.293813670191454,-0.10818066720577704,9.0703886554506>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5 }
    cylinder { m*<7.649165108191419,-0.19710094320013397,-5.509104634594742>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5}
    cylinder { m*<-5.515026912078784,4.545324619746131,-3.034220506926549>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5 }
    cylinder {  m*<-2.421157365789743,-3.4978359790755538,-1.4241447839691497>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5}

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
    sphere { m*<-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 1 }        
    sphere {  m*<0.293813670191454,-0.10818066720577704,9.0703886554506>, 1 }
    sphere {  m*<7.649165108191419,-0.19710094320013397,-5.509104634594742>, 1 }
    sphere {  m*<-5.515026912078784,4.545324619746131,-3.034220506926549>, 1}
    sphere { m*<-2.421157365789743,-3.4978359790755538,-1.4241447839691497>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.293813670191454,-0.10818066720577704,9.0703886554506>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5 }
    cylinder { m*<7.649165108191419,-0.19710094320013397,-5.509104634594742>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5}
    cylinder { m*<-5.515026912078784,4.545324619746131,-3.034220506926549>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5 }
    cylinder {  m*<-2.421157365789743,-3.4978359790755538,-1.4241447839691497>, <-1.146403163283668,-0.855826715639837,-0.7971487911022201>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    